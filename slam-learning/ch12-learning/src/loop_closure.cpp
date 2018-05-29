/*
 * Using DBoW to detect loop_closure
 */

#include <iostream>
#include <DBoW3/DBoW3.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    // Load the database and image
    DBoW3::Vocabulary vocab("../data/vocab_larger.yml.gz");
    if(vocab.empty())
    {
        cout << "Vocabulary does not exit" << endl;
        return 1;
    }
    cout << "Reading images..." << endl;
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
        waitKey(27);
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

    // we can compare the images directly or we can compare one image to a database images
    cout << "Comparing images with images" << endl;
    for (int i = 0; i < images.size(); ++i)
    {
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i], v1);
        for (int j = i; j < images.size(); ++j)
        {
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j],v2);
            double score = vocab.score(v1,v2);
            cout << "image " << i << " vs image " << j <<" : " << score << endl;
        }
        cout << endl;
    }

    // or with database
    cout << "Comparing images with database " << endl;
    DBoW3::Database db(vocab, false, 0);
    for (int k = 0; k < descriptors.size(); ++k)
    {
        db.add(descriptors[k]);
    }
    cout << "database info " << db << endl;
    for (int l = 0; l < descriptors.size(); ++l)
    {
        DBoW3::QueryResults ret;
        db.query(descriptors,ret,4);
        cout << "Searching for image " << l << " returns " << ret << endl << endl;
    }
    cout << "Done." << endl;

    return 0;
}