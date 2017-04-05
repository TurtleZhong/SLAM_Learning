#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <time.h>

using namespace std;
using namespace cv;
clock_t start,finish;

int main(int argc, char *argv[])
{
    Mat image = imread("/home/m/work/slam-learning/others/610_4.png");
    imshow("610", image);

    cv::initModule_nonfree();

    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;


    _detector = cv::FeatureDetector::create( "ORB" );
    //_descriptor = cv::DescriptorExtractor::create( "ORB" );

    vector<KeyPoint> kp1;
    Mat desp1;
    Mat feature;
    start = clock();
    _detector->detect(image, kp1);
    finish = clock();
    double time_orb = (double)(finish - start) / CLOCKS_PER_SEC;
    cout << "orb: " << kp1.size() << endl;
    cout << "ORB USE TIME: " << time_orb << "s" << endl;

    drawKeypoints(image, kp1, feature, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("keypoints", feature );
    imwrite("ORB.png", feature);
    waitKey(0);



    return 0;
}
