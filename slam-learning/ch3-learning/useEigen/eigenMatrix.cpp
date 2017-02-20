#include <iostream>
#include <ctime>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;

#define MATRIX_SIZE 50

int main()
{
//    Eigen::MatrixXd m = Eigen::MatrixXd::Random(3,3);
//    cout << m << endl;
    Eigen::Matrix<float,2,3> matrix_23;

    Eigen::Vector3d v_3d;
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    Eigen::MatrixXd matrix_x;

    matrix_23 << 1,2,3,4,5,6;
    cout << matrix_23 << endl;


    v_3d << 3, 2, 1;
    matrix_x = matrix_33.cast<double>() * v_3d;
    cout << matrix_x << endl;

    matrix_33 = Eigen::Matrix3d::Random();
    cout << matrix_33 << endl << endl;

    cout << matrix_33.transpose() << endl;
    cout << matrix_33.sum() << endl;
    cout << matrix_33.trace() << endl;
    cout << 10 * matrix_33 << endl;
    cout << matrix_33.inverse() << endl;
    cout << matrix_33.determinant() << endl;

    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
    Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd;
    v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock();
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time use in normal inverse is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;

    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time use in Qr compsition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;

    return 0;
}
