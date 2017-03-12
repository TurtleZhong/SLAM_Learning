#include <iostream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/features2d.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>



using namespace std;
using namespace cv;


int main(int argc, char** argv)
{


    /*
      * step1: load the original image;
      */
    Mat srcImage = imread("/home/m/workspace/projects/slam-learning/ch7-learning/1.png", 1);
    imshow("original image", srcImage);

    /*
      * define the detect param: use ORB
      */
    Ptr<ORB> orb = ORB::create();

    /*
      * use detect() function to detect keypoints
      */
    vector<KeyPoint> keyPoints;
    orb->detect(srcImage, keyPoints );

    /*
      * conpute the extractor and show the keypoints
      */
    Mat descriptors;
    orb->compute(srcImage, keyPoints, descriptors);


    /*
      * FLANN
      */
    flann::Index flannIndex(descriptors, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

    /*
      * Actually this is a sequence of image but this is just an example.
      */
    Mat testImage = imread("/home/m/workspace/projects/slam-learning/ch7-learning/2.png", 1);
    vector<KeyPoint> testKeyPoints;
    Mat testDescriptors;
    orb->detect(testImage, testKeyPoints);
    orb->compute(testImage, testKeyPoints, testDescriptors);

    /*Match the feature*/
    Mat matchIndex(testDescriptors.rows, 2, CV_32SC1);
    Mat matchDistance(testDescriptors.rows, 2, CV_32SC1);
    flannIndex.knnSearch(testDescriptors, matchIndex, matchDistance, 2, flann::SearchParams());

    vector<DMatch> goodMatches;
    for (int i = 0; i < matchDistance.rows; i++)
    {
        if(matchDistance.at<float>(i,0) < 0.6 * matchDistance.at<float>(i, 1))
        {
            DMatch dmatchs(i, matchIndex.at<int>(i,0), matchDistance.at<float>(i,1));
            goodMatches.push_back(dmatchs);
        }
    }

    Mat resultImage;
    drawMatches(testImage, testKeyPoints, srcImage, keyPoints, goodMatches, resultImage);
    imshow("result of Image", resultImage);

    cout << "We got " << goodMatches.size() << " good Matchs" << endl;
    cout << "OpenCV version: "
         << CV_MAJOR_VERSION << "."
         << CV_MINOR_VERSION << "."
         << CV_SUBMINOR_VERSION
         << std::endl;



    waitKey(0);



    return 0;
}
