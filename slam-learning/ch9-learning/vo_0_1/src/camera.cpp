/*
 * This code is used to coordinate transform
 *
 * world coordinate <---> camera coordinate <---> pixel coordinate
 *       X                        x                      u
 *       Y                        y                      v
 *       Z                        z(1)
 *
*/

#include "camera.h"

using namespace myslam;

Camera::Camera()
{

}

Vector3d Camera::world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3 &T_c_w)
{
    return T_c_w * p_w;
}

Vector3d Camera::camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3 &T_c_w)
{
    return T_c_w.inverse() * p_c;
}

Vector2d Camera::camera2pixel(const Eigen::Vector3d &p_c)
{
    /*
     * u = (fx * x)/ z + cx
     * v = (fy * y)/ z + cy
    */
    return Vector2d (
                fx_ * p_c(0,0) / p_c(2,0) + cx_,
                fy_ * p_c(1,0) / p_c(2,0) + cy_
                );
}

Vector3d Camera::pixel2camera(const Eigen::Vector2d &p_p, double depth)
{
    /*
     * x = (u - cx)*z / fx
     * v = (v - cy)*z / fy
    */
    return Vector3d (
                ( p_p(0,0) - cx_ ) * depth / fx_,
                ( p_p(1,0) - cy_ ) * depth / fy_,
                depth
                );
}

Vector2d Camera::world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3 &T_c_w)
{
    return camera2pixel( world2camera(p_w, T_c_w));
}

Vector3d Camera::pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3 &T_c_w, double depth)
{
    return camera2world( pixel2camera(p_p, depth), T_c_w);
}

