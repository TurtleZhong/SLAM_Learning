#include <iostream>
#include <ctime>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;

#define MATRIX_SIZE 50

int main()
{

    cout << "useEigen" << endl;
    Eigen::Matrix<float,2,3> matrix_23;
    matrix_23 << 1,2,3,4,5,6;
    cout << matrix_23 << endl;

    return 0;
}
