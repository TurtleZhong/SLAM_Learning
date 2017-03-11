#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(Mat& img1, Mat& img2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<DMatch>& matches);
void pose_estimation_2d2d(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
                          vector<DMatch> matches, Mat& R, Mat& t);

int main(int argc, char *argv[])
{

    Mat img1 = imread("../1.png", 1);
    Mat img2 = imread("../2.png" , 1);
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;

    find_feature_matches(img1, img2, keypoints1, keypoints2, matches);
    Mat R,t;
    pose_estimation_2d2d(keypoints1, keypoints2, matches, R, t);

    waitKey(0);
    return 0;
}

/*
 * input: image1,image2
 * output: keypoints1,keypoints2,goodmatches
 * function: detect the feature use ORB meathod
 */
void find_feature_matches(Mat& img1, Mat& img2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<DMatch>& matches)
{
    /*
      * step1: load the original image turn it into gray;
      */


    /*
      * define the detect param: use ORB
      */
    OrbFeatureDetector featureDetector;
    Mat descriptors;

    /*
      * use detect() function to detect keypoints
      */
    featureDetector.detect(img1, keypoints1);

    /*
      * conpute the extractor and show the keypoints
      */
    OrbDescriptorExtractor featureEvaluator;
    featureEvaluator.compute(img1, keypoints1, descriptors);

    /*
      * FLANN
      */
    flann::Index flannIndex(descriptors, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

    Mat testDescriptors;
    featureDetector.detect(img2, keypoints2);
    featureEvaluator.compute(img2, keypoints2,testDescriptors);

    /*Match the feature*/
    Mat matchIndex(testDescriptors.rows, 2, CV_32SC1);
    Mat matchDistance(testDescriptors.rows, 2, CV_32SC1);
    flannIndex.knnSearch(testDescriptors, matchIndex, matchDistance, 2, flann::SearchParams());

    //vector<DMatch> goodMatches;
    for (int i = 0; i < matchDistance.rows; i++)
    {
        if(matchDistance.at<float>(i,0) < 0.6 * matchDistance.at<float>(i, 1))
        {
            DMatch dmatchs(i, matchIndex.at<int>(i,0), matchDistance.at<float>(i,1));
            matches.push_back(dmatchs);
        }
    }

    Mat resultImage;
    drawMatches(img2, keypoints2, img1, keypoints1, matches, resultImage);
    imshow("result of Image", resultImage);

    cout << "We got " << matches.size() << " good Matchs" << endl;
}

void pose_estimation_2d2d(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
                          vector<DMatch> matches, Mat& R, Mat& t)
{
    /*
     * camera internal param
     */
    Mat K = (Mat_<double> (3,3) << 520.9, 0, 325.1, 521.0, 249.7, 0, 0, 1);
    vector<Point2f> points1, points2;

    for(int i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    /*
     * compute the F (fundmental) Matrix
     */
    Mat fundamentalMatrix;
    fundamentalMatrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "fundamental matrix is " << endl << fundamentalMatrix << endl;

    /*
     * compute the E (essential) Matrix opencv2.4.11 could not find this function
     */

    /*
     * compute the homography Matrix
     */
    Mat homographyMatrix;
    homographyMatrix = findHomography(points1, points2);
    cout << "homography Matrix is " << endl << homographyMatrix << endl;


}
