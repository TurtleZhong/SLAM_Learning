#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

using namespace std;

int main()
{
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    Eigen::AngleAxisd rotation_vector (M_PI/4, Eigen::Vector3d(0,0,1)); //the vector must be normalized!!

    cout.precision(3);
    cout << "rotation matrix = \n" << rotation_vector.toRotationMatrix() << endl;
    rotation_matrix = rotation_vector.toRotationMatrix();
    // use angleAxisd to rotation

    Eigen::Vector3d v (1,0,0);
    Eigen::Vector3d v_rotated = rotation_vector * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;

    //use rotation_matrix to totation
    v_rotated = rotation_matrix * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;

    //euler_angles
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0); //ZYX
    cout << "yaw pitch roll = \n" << euler_angles.transpose() << endl;

    //use Isometry

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(rotation_vector);
    T.pretranslate (Eigen::Vector3d(1,3,4));
    cout << "Transform matrix = \n" << T.matrix() << endl;

    //use T to
    Eigen::Vector3d v_transformed = T * v;
    cout << "v_transformed = \n " << v_transformed.transpose() << endl;

    //quaternion

    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    cout << "quaternion = \n" << q.coeffs() << endl;
    v_rotated = q * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;


    return 0;
}
