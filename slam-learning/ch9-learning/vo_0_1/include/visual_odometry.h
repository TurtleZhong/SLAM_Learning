#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H

#include "common_include.h"
#include "map.h"

namespace myslam
{
class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;        /*shared ptr*/
    enum VOstate {
        INITIALIZING = -1,
        OK = 0,
        LOST = 1
    };

    /*data members are below*/
    VOstate                 state_;                /*current vo status*/
    Map::Ptr                map_;                  /*map with all frames and mappoints*/
    Frame::Ptr              ref_;                  /*reference frame*/
    Frame::Ptr              curr_;                 /*current frame*/

    cv::Ptr<cv::ORB>        orb_;                  /*orb detector and computer*/
    vector<cv::Point3f>     pts_3d_ref_;           /*3d points in reference frame*/
    vector<cv::KeyPoint>    keypoints_curr_;       /*keypoints in current frame*/

    Mat                     descriptors_curr_;     /*descriptor in current frame*/
    Mat                     descriptors_ref_;      /*descriptor in reference frame*/
    vector<cv::DMatch>      feature_matches_;      /**/
    cv::FlannBasedMatcher   matcher_flann_;        /* flann matcher */
    SE3                     T_c_r_estimated_;      /*the estimated pose of current frame*/
    int                     num_inliers_;          /*number of inlier features */
    int                     num_lost_;             /*number of lost times*/

    int                     num_of_features_;      /*number of features*/
    int                     scale_factor_;         /*scale in image pyramid*/
    int                     level_pyramid_;        /*number of pyramid levels*/
    float                   match_ratio_;          /*ratio for selecting good matches*/
    int                     max_num_lost_;         /*max number of continuos lost times*/
    int                     min_inliers_;          /*minimum inliers*/

    double                  key_frame_min_rot_;    /*min rotation of two keyframes*/
    double                  key_frame_min_trans_;  /*min translation of two keyframes*/

public:
    VisualOdometry();
    ~VisualOdometry();

    bool addFrame( Frame::Ptr frame );

    /*functions are below*/
protected:

    void extractKeyPoints();
    void computeDescriptors();
    void featureMatching();
    void poseEstimationPnP();
    void setRef3DPoints();

    void addKeyFrame();
    bool checkEstimatedPose();
    bool checkKeyFrame();

};


}


#endif // VISUAL_ODOMETRY_H
