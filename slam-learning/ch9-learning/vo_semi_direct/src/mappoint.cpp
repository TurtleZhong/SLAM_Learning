#include "common_include.h"
#include "mappoint.h"

using namespace myslam;

MapPoint::MapPoint()
    : id_(-1), pos_(Vector3d(0,0,0)), norm_(Vector3d(0,0,0)), good_(true), visible_times_(0), matched_times_(0)
{

}


MapPoint::MapPoint (long id, const Vector3d& position, const Vector3d& norm, Frame* frame, float gray_value)
: id_(id), pos_(position), norm_(norm), good_(true), visible_times_(1), matched_times_(1), gray_value_(gray_value)
{
    observed_frames_.push_back(frame);
}

MapPoint::Ptr MapPoint::createMapPoint()
{
    return MapPoint::Ptr(
        new MapPoint( factory_id_++, Vector3d(0,0,0), Vector3d(0,0,0) )
    );
}


MapPoint::Ptr MapPoint::createMapPoint(const Eigen::Vector3d &pos_world,
                                       const Eigen::Vector3d &norm_,
                                       float gray_value,
                                       Frame *frame)
{

    return MapPoint::Ptr(
                new MapPoint(factory_id_++, pos_world, norm_, frame, gray_value)
                );
}

unsigned long MapPoint::factory_id_ = 0;
