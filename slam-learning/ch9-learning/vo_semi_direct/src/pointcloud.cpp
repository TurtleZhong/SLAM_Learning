#include "pointcloud.h"

using namespace myslam;



Point_Cloud::Point_Cloud()
{
    this->voxel_grid_           = Config::get<float>("voxel_grid");
    this->camera_->fx_          = Config::get<float>("camera.fx");
    this->camera_->fy_          = Config::get<float>("camera.fy");
    this->camera_->cx_          = Config::get<float>("camera.cx");
    this->camera_->cy_          = Config::get<float>("camera.cy");
    this->camera_->depth_scale_ = Config::get<float>("camera.depth_scale");
}

PointCloud::Ptr Point_Cloud::image2PointCloud(cv::Mat color, cv::Mat depth)
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
            p.z = double(d) / camera_->depth_scale_;
            p.x = (n - camera_->cx_) * p.z / camera_->fx_;
            p.y = (m - camera_->cy_) * p.z / camera_->fx_;

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


PointCloud::Ptr Point_Cloud::joinPointCloud(PointCloud::Ptr original, Frame::Ptr newFrame, SE3 T)
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
