#ifndef POINT_CLOUD_H
#define POINT_CLOUD_H

#include "common_include.h"
#include "frame.h"
#include "camera.h"
#include "config.h"

namespace myslam {

class Point_Cloud
{
public:
    typedef shared_ptr<Point_Cloud> Ptr;
    Camera::Ptr  camera_;
    float        voxel_grid_;

public:
    Point_Cloud();

    PointCloud::Ptr image2PointCloud(cv::Mat color, cv::Mat depth);


    // 输入：原始点云，新来的帧以及它的位姿
    // 输出：将新来帧加到原始帧后的图像
    PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, Frame::Ptr newFrame, SE3 T);



};

}







#endif // POINTCLOUD_H
