

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "config.h"
#include "visual_odometry.h"
#include "g2o_types.h"
#include "g2o_direct.h"
#include "pointcloud.h"


namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ),
    ref_ ( nullptr ),
    curr_ ( nullptr ),
    map_ ( new Map ),
    num_lost_ ( 0 ),
    num_inliers_ ( 0 ),
    matcher_flann_ ( new cv::flann::LshIndexParams ( 5,10,2 )  )
{
    num_of_features_         = Config::get<int> ( "number_of_features" );
    scale_factor_            = Config::get<double> ( "scale_factor" );
    level_pyramid_           = Config::get<int> ( "level_pyramid" );
    match_ratio_             = Config::get<float> ( "match_ratio" );
    max_num_lost_            = Config::get<float> ( "max_num_lost" );
    min_inliers_             = Config::get<int> ( "min_inliers" );
    key_frame_min_rot        = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans      = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_   = Config::get<double> ( "map_point_erase_ratio" );
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
    voxel_grid_              = Config::get<float>("voxel_grid");
    //viewer_ = pcl::visualization::CloudViewer ("viewer");
    //pcl::visualization::CloudViewer viewer_tmp("viewer");



}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = frame;
        // extract features from first frame and add them into map
        extractKeyPoints();
        computeDescriptors();
        addKeyFrame();      // the first frame is a key-frame
        break;
    }
    case OK:
    {
        curr_ = frame;
        curr_->T_c_w_ = ref_->T_c_w_;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_w_estimated_;
            optimizeMap();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();

            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

/*
 * Direct Method
 *
 */
bool VisualOdometry::addFrame(Frame::Ptr frame, string Method)
{
    /*
     * Direct Method
     *
     */
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = frame;
        cloud_ = image2PointCloud( ref_->color_, ref_->depth_ );
        cout <<"***add pointcloud!!***" << endl;
        // extract features from first frame and add them into map
        extractGradiantsPoints();
        addKeyFrame();      // the first frame is a key-frame
        break;
    }
    case OK:
    {
        curr_ = frame;
        cout <<  "curr_->id_: " << curr_->id_ << endl;

        poseEstimationDirect();
        curr_->T_c_w_ = T_c_w_estimated_;
        if ( checkKeyFrame() == true ) // is a key-frame
        {
            cout << "/*************add a keyframe!************/" << endl;
            addKeyFrame();


            cloud_ = joinPointCloud( cloud_, ref_, ref_->T_c_w_ );
            //pcl::visualization::CloudViewer viewer("viewer");
            //viewer.showCloud( cloud_ );

            if(curr_->id_ % 10 == 0)
            {
                pcl::io::savePCDFile( "/home/m/work/slam-learning/ch9-learning/vo_semi_direct/results/pcd/result.pcd", *cloud_ );
            }

            measurements_curr_.clear();  /*we need to clear the measurement points cuz update the key-frame*/
            extractGradiantsPoints(); /*cuz update the keyframe so we need to update the gradiants points*/

        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

void VisualOdometry::extractKeyPoints()
{
    boost::timer timer;
    orb_->detect ( curr_->color_, keypoints_curr_ );
    cout<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::computeDescriptors()
{
    boost::timer timer;
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
    cout<<"descriptor computation cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::featureMatching()
{
    boost::timer timer;
    vector<cv::DMatch> matches;
    // select the candidates in map
    Mat desp_map;
    vector<MapPoint::Ptr> candidate;
    for ( auto& allpoints: map_->map_points_ )
    {
        MapPoint::Ptr& p = allpoints.second;
        // check if p in curr frame image
        if ( curr_->isInFrame(p->pos_) )
        {
            // add to candidate
            p->visible_times_++;
            candidate.push_back( p );
            desp_map.push_back( p->descriptor_ );
        }
    }

    matcher_flann_.match ( desp_map, descriptors_curr_, matches );
    // select the best matches
    float min_dis = std::min_element (
                matches.begin(), matches.end(),
                [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;

    match_3dpts_.clear();
    match_2dkp_index_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            match_3dpts_.push_back( candidate[m.queryIdx] );
            match_2dkp_index_.push_back( m.trainIdx );
        }
    }
    cout<<"good matches: "<<match_3dpts_.size() <<endl;
    cout<<"match cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;

    for ( int index:match_2dkp_index_ )
    {
        pts2d.push_back ( keypoints_curr_[index].pt );
    }
    for ( MapPoint::Ptr pt:match_3dpts_ )
    {
        pts3d.push_back( pt->getPositionCV() );
    }

    Mat K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
              );
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac ( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    T_c_w_estimated_ = SE3 (
                SO3 ( rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ) ),
                Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) )
                );

    // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block ( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
                            T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()
                            ));
    optimizer.addVertex ( pose );

    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int> ( i,0 );
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId ( i );
        edge->setVertex ( 0, pose );
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Vector3d ( pts3d[index].x, pts3d[index].y, pts3d[index].z );
        edge->setMeasurement ( Vector2d ( pts2d[index].x, pts2d[index].y ) );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        // set the inlier map points
        match_3dpts_[index]->matched_times_++;
    }

    optimizer.initializeOptimization();
    optimizer.optimize ( 10 );

    T_c_w_estimated_ = SE3 (
                pose->estimate().rotation(),
                pose->estimate().translation()
                );

    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    // if the motion is too large, it is probably wrong
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm() <<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    cout << "ref_->T_c_w_ = "<< endl <<  ref_->T_c_w_.matrix()<< endl;
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    if ( map_->keyframes_.empty() )
    {
        // first key-frame, add all 3d points into map
        for ( int i=0; i<measurements_curr_.size(); i++ )
        {
            double d = ref_->findDepth ( measurements_curr_[i].gradiant_points.x, measurements_curr_[i].gradiant_points.y );
            if ( d<0 )
                continue;
            Vector3d p_world = ref_->camera_->pixel2world (
                        Vector2d ( measurements_curr_[i].gradiant_points.x, measurements_curr_[i].gradiant_points.y ),
                        curr_->T_c_w_, d
                        );
            Vector3d n = p_world - ref_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                        p_world, n, measurements_curr_[i].grayscale, curr_.get()
                        );
            map_->insertMapPoint( map_point );
        }
    }

    map_->insertKeyFrame ( curr_ );
    ref_ = curr_;
}

void VisualOdometry::addMapPoints()
{
    // add the new map points into map
    for ( int i=0; i<measurements_curr_.size(); i++ )
    {
        double d = ref_->findDepth ( measurements_curr_[i].gradiant_points.x, measurements_curr_[i].gradiant_points.y );
        if ( d<0 )
            continue;
        Vector3d p_world = ref_->camera_->pixel2world (
                    Vector2d ( measurements_curr_[i].gradiant_points.x, measurements_curr_[i].gradiant_points.y ),
                    curr_->T_c_w_, d
                    );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(
                    p_world, n, measurements_curr_[i].grayscale, curr_.get()
                    );
        map_->insertMapPoint( map_point );
    }
}

void VisualOdometry::optimizeMap()
{
    // remove the hardly seen and no visible points
    for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )
    {
        if ( !curr_->isInFrame(iter->second->pos_) )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        float match_ratio = float(iter->second->matched_times_)/iter->second->visible_times_;
        if ( match_ratio < map_point_erase_ratio_ )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }

        double angle = getViewAngle( curr_, iter->second );
        if ( angle > M_PI/6. )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        if ( iter->second->good_ == false )
        {
            // TODO try triangulate this map point
        }
        iter++;
    }

    if ( match_2dkp_index_.size()<100 )
        addMapPoints();
    if ( map_->map_points_.size() > 1000 )
    {
        // TODO map is too large, remove some one
        map_point_erase_ratio_ += 0.05;
    }
    else
        map_point_erase_ratio_ = 0.1;
    cout<<"map points: "<<map_->map_points_.size()<<endl;
}

double VisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
{
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

void VisualOdometry::extractGradiantsPoints()
{
    Mat gray;
    cv::cvtColor ( curr_->color_, gray, cv::COLOR_BGR2GRAY );
    // select the pixels with high gradiants
    for ( int x=10; x<curr_->color_.cols-10; x++ )
    {
        for ( int y=10; y<curr_->color_.rows-10; y++ )
        {
            Eigen::Vector2d delta (
                        gray.ptr<uchar>(y)[x+1] - gray.ptr<uchar>(y)[x-1],
                    gray.ptr<uchar>(y+1)[x] - gray.ptr<uchar>(y-1)[x]
                    );
            if ( delta.norm() < 75 )
                continue;
            ushort d = curr_->depth_.ptr<ushort> (y)[x];
            if ( d==0 )
                continue;
            if ( double(d/curr_->camera_->depth_scale_) > 6.0)
                continue; /*cuz the depth is too large so we can't use it*/
            //Eigen::Vector3d p3d = project2Dto3D ( x, y, d, fx, fy, cx, cy, depth_scale );
            Eigen::Vector3d p3d = curr_->camera_->pixel2camera(Vector2d(x,y), d);
            //cout << "p3d_points = " << p3d << endl;
            float grayscale = float ( gray.ptr<uchar> (y) [x] );
            measurements_curr_.push_back ( Measurement ( p3d, cv::Point2f(x,y), grayscale ) );
            //gradiants_points_curr_.push_back( cv::Point2f(x,y));
        }
    }
    cout << "gradiantsPoints size is " << measurements_curr_.size() << endl;
}

void VisualOdometry::poseEstimationDirect()
{
    Frame tmp = *curr_; /*5.27 by zhong*/
    //    cv::imshow("tmp", tmp.color_);
    //    cv::waitKey(0);
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;  // 求解的向量是6＊1的
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();
    DirectBlock* solver_ptr = new DirectBlock ( linearSolver );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr ); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );
    optimizer.setVerbose( false ); /*debug information*/

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate ( g2o::SE3Quat ( T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation() ) );
    pose->setId ( 0 );
    optimizer.addVertex ( pose );

    // 添加边
    int id=1;
    for ( Measurement m: measurements_curr_ )
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect (
                    m.pos_world,
                    tmp
                    );
        edge->setVertex ( 0, pose );
        edge->setMeasurement ( m.grayscale );
        edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );
        edge->setId ( id++ );
        optimizer.addEdge ( edge );
    }
    //cout<<"edges in graph: "<<optimizer.edges().size() <<endl;
    optimizer.initializeOptimization();
    optimizer.optimize ( 15 );
    T_c_w_estimated_ = SE3 (
                pose->estimate().rotation(),
                pose->estimate().translation()
                );
    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;

    // plot the feature points
    cv::Mat img_show ( curr_->color_.rows*2, curr_->color_.cols, CV_8UC3 );
    ref_->color_.copyTo ( img_show ( cv::Rect ( 0,0,curr_->color_.cols, curr_->color_.rows ) ) );
    curr_->color_.copyTo ( img_show ( cv::Rect ( 0,curr_->color_.rows,curr_->color_.cols, curr_->color_.rows ) ) );
    for ( Measurement m:measurements_curr_ )
    {
        if ( rand() > RAND_MAX/5 )
            continue;
        Eigen::Vector3d p = m.pos_world;
        //Eigen::Vector2d pixel_prev = project3Dto2D ( p ( 0,0 ), p ( 1,0 ), p ( 2,0 ), fx, fy, cx, cy );
        Eigen::Vector2d pixel_prev = curr_->camera_->camera2pixel(p);
        Eigen::Vector3d p2 = T_c_w_estimated_*m.pos_world;
        //Eigen::Vector2d pixel_now = project3Dto2D ( p2 ( 0,0 ), p2 ( 1,0 ), p2 ( 2,0 ), fx, fy, cx, cy );
        Eigen::Vector2d pixel_now = curr_->camera_->camera2pixel(p2);
        if ( pixel_now(0,0)<0 || pixel_now(0,0)>=curr_->color_.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=curr_->color_.rows )
            continue;

        float b = 0;
        float g = 250;
        float r = 0;
        img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3] = b;
        img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3+1] = g;
        img_show.ptr<uchar>( pixel_prev(1,0) )[int(pixel_prev(0,0))*3+2] = r;

        img_show.ptr<uchar>( pixel_now(1,0)+curr_->color_.rows )[int(pixel_now(0,0))*3] = b;
        img_show.ptr<uchar>( pixel_now(1,0)+curr_->color_.rows )[int(pixel_now(0,0))*3+1] = g;
        img_show.ptr<uchar>( pixel_now(1,0)+curr_->color_.rows )[int(pixel_now(0,0))*3+2] = r;
        cv::circle ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 4, cv::Scalar ( b,g,r ), 2 );
        cv::circle ( img_show, cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +curr_->color_.rows ), 4, cv::Scalar ( b,g,r ), 2 );
    }
    cv::imshow ( "result", img_show );
    //cv::waitKey ( 27 );
}

PointCloud::Ptr VisualOdometry::image2PointCloud(cv::Mat color, cv::Mat depth)
{
    PointCloud::Ptr cloud ( new PointCloud );

    for (int m = 0; m < depth.rows; m+=1)
        for (int n=0; n < depth.cols; n+=1)
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) / ref_->camera_->depth_scale_;
            p.x = (n - ref_->camera_->cx_) * p.z / ref_->camera_->fx_;
            p.y = (m - ref_->camera_->cy_) * p.z / ref_->camera_->fx_;

            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            cloud->points.push_back( p );
        }
    // 设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}


PointCloud::Ptr VisualOdometry::joinPointCloud(PointCloud::Ptr original, Frame::Ptr newFrame, SE3 T)
{

    PointCloud::Ptr newCloud = image2PointCloud( newFrame->color_, newFrame->depth_ );

    // 合并点云
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *original, *output, T.matrix() );
    *newCloud += *output;

    // Voxel grid 滤波降采样
    static pcl::VoxelGrid<PointT> voxel;
    float gridsize = voxel_grid_;
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    voxel.setInputCloud( newCloud );
    PointCloud::Ptr tmp( new PointCloud() );
    voxel.filter( *tmp );
    return tmp;
}





}
