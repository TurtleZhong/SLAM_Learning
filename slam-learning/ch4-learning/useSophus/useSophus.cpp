#include <iostream>

//add Eigen libraries
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>


//add Sophus libraries
#include <sophus/so3.h>
#include <sophus/se3.h>


using namespace std;
int main()
{
    Eigen::AngleAxisd rotation_vector = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1));
    Eigen::Matrix3d R = rotation_vector.matrix();
    cout << "rotation matrix = \n" << R << endl;

    Sophus::SO3 SO3_R(R);                //using rotation matrix to creat SO3
    Sophus::SO3 SO3_v(0, 0, M_PI/2);     //using rotation vector to creat SO3
    Eigen::Quaterniond q(R);
    Sophus::SO3 SO3_q(q);                //using quaternion to creat SO3

    cout << "SO3 from matrix: " << SO3_R << endl;
    cout << "SO3 from vector: " << SO3_v << endl;
    cout << "SO3 from quaternion: " << SO3_q << endl;

    //use log to map SO3 to so3

    Eigen::Vector3d so3 = SO3_R.log();
    cout << "so3 = " << so3.transpose() << endl;

    //hat
    cout << "so3 hat = \n" << Sophus::SO3::hat(so3) << endl;
    //vee
    cout << "so3 hat vee = \n" << Sophus::SO3::vee(Sophus::SO3::hat(so3));

    //bundle model update
    Eigen::Vector3d update_so3(1e-4, 0, 0);
    Sophus::SO3 SO3_update = Sophus::SO3::exp(update_so3) * SO3_R;
    cout << "\n SO3 update = \n" << SO3_update << endl;

    cout << "***********SE3*************" << endl;
    Eigen::Vector3d t(1,0,0);
    Sophus::SE3 SE3_Rt(R, t);
    Sophus::SE3 SE3_qt(q,t);
    cout << "SE3 from R,t = \n" << SE3_Rt << endl;
    cout << "SE3 from q,t = \n" << SE3_qt << endl;

    //li se3
    typedef Eigen::Matrix<double,6,1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    cout << "se3 = \n" << se3.transpose() << endl;

    cout << "se3 hat \n" << Sophus::SE3::hat(se3) << endl;

    Vector6d update_se3;
    update_se3.setZero();
    update_se3(0,0) = double(1e-4);
    Sophus::SE3 SE3_updated = Sophus::SE3::exp(update_se3) * SE3_Rt;
    cout << "SE3 updated = \n" << SE3_updated.matrix() << endl;




    return 0;
}
