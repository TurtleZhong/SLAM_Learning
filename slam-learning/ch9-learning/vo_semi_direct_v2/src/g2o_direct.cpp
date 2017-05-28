#include "g2o_direct.h"

using namespace myslam;

void EdgeSE3ProjectDirect::computeError()
{
    const VertexSE3Expmap* v  =static_cast<const VertexSE3Expmap*> ( _vertices[0] );
    Eigen::Vector3d x_local = v->estimate().map ( p_world_ );
    float x = x_local[0]*frame_.camera_->fx_/x_local[2] + frame_.camera_->cx_;
    float y = x_local[1]*frame_.camera_->fy_/x_local[2] + frame_.camera_->cy_;
    // check x,y is in the image
    if ( x-4<0 || ( x+4 ) >frame_.color_.cols || ( y-4 ) <0 || ( y+4 ) >frame_.color_.rows )
    {
        _error ( 0,0 ) = 0.0;
        this->setLevel ( 1 );
    }
    else
    {
        //_error ( 0,0 ) = getPixelValue ( x,y ) - _measurement; /*here must promise the getPixelValue is right*/
        _error ( 0,0 ) = frame_.findDepth(x,y) - _measurement;
    }
}

void EdgeSE3ProjectDirect::linearizeOplus()
{
    if ( level() == 1 )
    {
        _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
        return;
    }
    VertexSE3Expmap* vtx = static_cast<VertexSE3Expmap*> ( _vertices[0] );
    Eigen::Vector3d xyz_trans = vtx->estimate().map ( p_world_ );   // q in book

    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double invz = 1.0/xyz_trans[2];
    double invz_2 = invz*invz;

    float u = x*frame_.camera_->fx_*invz + frame_.camera_->cx_;
    float v = y*frame_.camera_->fy_*invz + frame_.camera_->cy_;

    // jacobian from se3 to u,v
    // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
    Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

    jacobian_uv_ksai ( 0,0 ) = - x*y*invz_2 *frame_.camera_->fx_;
    jacobian_uv_ksai ( 0,1 ) = ( 1+ ( x*x*invz_2 ) ) *frame_.camera_->fx_;
    jacobian_uv_ksai ( 0,2 ) = - y*invz *frame_.camera_->fx_;
    jacobian_uv_ksai ( 0,3 ) = invz *frame_.camera_->fx_;
    jacobian_uv_ksai ( 0,4 ) = 0;
    jacobian_uv_ksai ( 0,5 ) = -x*invz_2 *frame_.camera_->fx_;

    jacobian_uv_ksai ( 1,0 ) = - ( 1+y*y*invz_2 ) *frame_.camera_->fy_;
    jacobian_uv_ksai ( 1,1 ) = x*y*invz_2 *frame_.camera_->fy_;
    jacobian_uv_ksai ( 1,2 ) = x*invz *frame_.camera_->fy_;
    jacobian_uv_ksai ( 1,3 ) = 0;
    jacobian_uv_ksai ( 1,4 ) = invz *frame_.camera_->fy_;
    jacobian_uv_ksai ( 1,5 ) = -y*invz_2 *frame_.camera_->fy_;

    Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

    jacobian_pixel_uv ( 0,0 ) = ( getPixelValue ( u+1,v )-getPixelValue ( u-1,v ) ) /2;
    jacobian_pixel_uv ( 0,1 ) = ( getPixelValue ( u,v+1 )-getPixelValue ( u,v-1 ) ) /2;

    _jacobianOplusXi = jacobian_pixel_uv*jacobian_uv_ksai;
}
