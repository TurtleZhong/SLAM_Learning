#ifndef G2O_DIRECT_H
#define G2O_DIRECT_H

#include "common_include.h"
#include "camera.h"
#include "frame.h"

/*g2o dependencies*/
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>

using namespace std;
using namespace g2o;

/*define the */
namespace myslam
{

// 一次测量的值，包括一个世界坐标系下三维点与一个灰度值
struct Measurement
{
    Measurement ( Eigen::Vector3d p, float g ) : pos_world ( p ), grayscale ( g ) {}
    Eigen::Vector3d pos_world;
    float grayscale;
};


class EdgeSE3ProjectDirect: public BaseUnaryEdge<1, double, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect() {}

    EdgeSE3ProjectDirect ( Eigen::Vector3d p_world, Camera* camera, Frame* frame  )
        : p_world_ ( p_world ), camera_(camera), frame_( frame )
    {}

    virtual void computeError();
    virtual void linearizeOplus();

    virtual bool read( std::istream& in ){}
    virtual bool write(std::ostream& os) const {}

    Eigen::Vector3d    p_world_;  /*word points*/
    Camera*            camera_;   /*params of camera*/
    Frame::Ptr         frame_;    /*gray and depth*/

protected:
    // get a gray scale value from reference image (bilinear interpolated双线性插值)
    inline float getPixelValue ( float x, float y )
    {
        uchar *data =  & frame_->color_.data[ int ( y ) * frame_->color_.step + int ( x ) ];
        float xx = x - floor ( x );
        float yy = y - floor ( y );
        return float (
                    ( 1-xx ) * ( 1-yy ) * data[0] +
                xx* ( 1-yy ) * data[1] +
                ( 1-xx ) *yy*data[ frame_->color_.step ] +
                xx*yy*data[frame_->color_.step+1]
                );
    }

};

}





#endif // G2O_DIRECT_H
