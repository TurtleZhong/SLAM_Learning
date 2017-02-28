#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;
using namespace cv;

/*
 * model: f(x) = exp(ax^2 + bx +c)
 * noise: Gaussian noise
 * step1: Generate the x - y of the model
 * step2: Add noise to the model function
 * step3: Use Ceres to solve the problem
 */


struct CURVE_FITTING_COST
{

    CURVE_FITTING_COST (double x, double y ): _x(x), _y(y) {}
    //ResidualBlock
    template <typename T >
    bool operator() (const T* const abc, T* residual ) const
    {
        //y - exp(ax^2 + bx + c)
        residual[0] = T(_y) - ceres::exp( abc[0]* T(_x) * T(_x) + abc[1]*T(_x) + abc[2]);
        return true;
    }
    const double _x, _y;

};

int main(int argc, char** argv)
{

    double a = 1.0, b = 2.0, c = 3.0;  //The true param of model
    int N = 100;
    double w_sigma = 1.0;
    RNG rng;
    double abc[3] = {0,0,0};

    vector<double> x_data, y_data;

    cout << "generating data: " << endl;
    for (int i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back( exp( a*x*x + b * x + c ) + rng.gaussian( w_sigma ) );
        cout << x_data[i] << " " << y_data[i] << endl;
    }

    //solve the problem
    ceres::Problem problem;
    for(int i = 0; i < N; i++)
    {
        problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,3> (
                        new CURVE_FITTING_COST (x_data[i], y_data[i] )
                        ),
                    nullptr,
                    abc
                    );
    }

    //
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now(); // time start

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 -t1);

    cout << "solve time cost = " << time_used.count() << " seconds." << endl;
    cout << summary.BriefReport() << endl;





    return 0;
}
