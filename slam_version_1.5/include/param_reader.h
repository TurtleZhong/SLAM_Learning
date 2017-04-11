#ifndef PARAM_READER_H
#define PARAM_READER_H

#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <map>

using namespace std;
const string param_file_path = "../parameters.txt";

class ParameterReader
{
public:
    ParameterReader();

    string getData(const string& key);


    map<string, string> data;

protected:

    string detector;
    string descriptor;

    int good_match_threshold;

    /*camera parameters*/
    double cx;
    double cy;
    double fx;
    double fy;
    double scale;

    int start_index;
    int end_index;

    string use_tum;
    string tum_dataset_path;
    string rgb_dir;
    string rgb_extension;
    string depth_dir;
    string depth_extension;

    double voxel_grid;

    string visualize_pointcloud;

    int min_good_match;

    int min_inliers;

    double max_norm;

    double keyframe_threshold;
    double max_norm_lp;

    string check_loop_closure;
    int nearby_loops;
    int random_loops;

};

//全局变量指针
//extern ParameterReader* Ptr_ParamReader;

#endif // PARAM_READER_H
