#include <iostream>
#include <fstream>
#include <list>
#include <vector>

using namespace std;
#include <opencv2/opencv.hpp>
using namespace cv;


int main(int argc, char** argv)
{
    string path_of_dataset = "/home/m/work/slam-learning/ch8-learning/data";
    string associate_file = path_of_dataset + "/associate.txt";
    ifstream fin(associate_file);

    string rgb_file, depth_file, time_rgb, time_depth;
    list<cv::Point2f> keypoints;
    Mat color, depth, last_color;

    for(int index = 0; index < 100; index++)
    {
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        color = imread(path_of_dataset + "/" + rgb_file);
        depth = imread(path_of_dataset + "/" + depth_file, -1);
        if(index == 0)
        {
            //FAST feature
            vector<KeyPoint> kps;
            Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
            detector->detect(color, kps);
            for(auto kp:kps)
            {
                keypoints.push_back(kp.pt);
            }
            last_color = color;
            continue;
        }
        if(color.data == nullptr || depth.data == nullptr)
            continue;

        //use LK to track
        vector<Point2f> next_keypoints;
        vector<Point2f> prev_keypoints;
        for(auto kp:keypoints)
        {
            prev_keypoints.push_back(kp);
        }
        vector<unsigned char> status;
        vector<float> error;
        calcOpticalFlowPyrLK(last_color, color, prev_keypoints, next_keypoints, status, error);

        //delete the lost keypoints;
        int i = 0;
        for(auto iter = keypoints.begin(); iter!= keypoints.end(); i++)
        {
            if(status[i] == 0)
            {
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = next_keypoints[i];
            iter++;
        }
        cout << "tracked keypoints: " << keypoints.size() << endl;
        if(keypoints.size() == 0)
        {
            cout << "all keypoints are lost " << endl;
            break;
        }
        Mat img_show = color.clone();
        for(auto kp: keypoints)
            circle(img_show, kp, 6, Scalar(0, 240, 0), 1);
        imshow("corners", img_show);
        waitKey(0);
        last_color = color;
    }



    return 0;
}
