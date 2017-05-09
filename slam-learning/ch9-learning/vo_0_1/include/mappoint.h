#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "common_include.h"

namespace myslam
{

class Frame;
class MapPoint
{
public:
    typedef shared_ptr<MapPoint>  Ptr;
    unsigned long                 id_;             /*ID*/
    Vector3d                      pos_;            /*Position in world*/
    Vector3d                      norm_;           /*Normal of viewing direction*/
    Mat                           descriptor_;     /*Descriptor for matching*/
    int                           observed_times_; /*being observed by feature matching*/
    int                           correct_times_;  /*being inliner in pos estimation*/

public:
    MapPoint();
    MapPoint(
            long id,
             Vector3d position,
             Vector3d norm
            );
    static MapPoint::Ptr createMapPoint();

};

}


#endif // MAPPOINT_H
