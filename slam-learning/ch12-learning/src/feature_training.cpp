/**
 * Traing the vocabulary with images
 */
#include <DBoW3/DBoW3.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;

int main()
{
    /**
     * read image from the ../data/image.txt
     */
    cout << "Reading imags..." << endl;
    vector<Mat> images;
    ifstream inFile("../data/image.txt");
    string root_path = "../data/";
    string tmp_path;
    while (inFile.good() && images.size() < 10)
    {
        inFile >> tmp_path;
        inFile.get();
        Mat image = imread(root_path+tmp_path, CV_LOAD_IMAGE_COLOR);
        images.push_back(image);
        imshow("image",image);
        waitKey(0);
    }
    cout << "Reading image suscessfully!" << endl;

    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;
    Ptr< Feature2D > detector = ORB::create();
    vector<Mat> descriptors;
    for ( Mat& image:images )
    {
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
    }

    // create vocabulary
    cout<<"creating vocabulary ... "<<endl;
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors );
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save( "../data/vocabulary.yml.gz" );
    cout<<"done"<<endl;

    return 0;
}

