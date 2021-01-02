#include <iomanip>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <glob.h>

using namespace std;
using namespace cv;

vector<string> get_file_path(string input_dir) {
    glob_t globbuf;
    vector<string> files;
    auto glob_files = [&](const string& suffix) {
        glob((input_dir + "/*." + suffix).c_str(), 0, NULL, &globbuf);
        for (size_t i = 0; i < globbuf.gl_pathc; i++) {
            files.push_back(globbuf.gl_pathv[i]);
        }
        globfree(&globbuf);
    };
    glob_files("png");
    return files;
}

string get_basename(string path){
    string basename = path.substr(path.find_last_of('/') + 1);
    basename = basename.substr(0, basename.find_last_of('.'));
    return basename;
}

void process(const string& img_path) {
    Mat img  = imread(img_path);
    Mat colimg = Mat::zeros(img.size(), CV_8UC3);
    for(int y = 0; y < img.rows; y++){
        for (int x = 0; x < img.cols; x++) {
            cv::Vec<unsigned char, 3> pixlabel = img.ptr<cv::Vec3b>(y)[x];
            int index = pixlabel[0];
            Vec3b out;
            if(index == 0){
                out = Vec3b({142, 47, 69});
            }else if(index == 1){
                out = Vec3b({0, 0, 255});
            }else if(index == 2){
                out = Vec3b({0, 255, 255});
            }else if(index==3){
                out = Vec3b({255, 0, 0});
            }else{
                out = Vec3b({0, 0, 0});
            }
            colimg.ptr<cv::Vec3b>(y)[x] = out;
        }
    }
    string basename = img_path.substr(img_path.size()-12, 12);
    cout << basename << endl;
    imwrite("../label/" + basename, colimg);
}

int main(int argc, char *argv[]) {
    string img_path = "../results/";
    vector<string> img_files = get_file_path(img_path);
    for (size_t i = 0; i < img_files.size(); i++) {
        process(img_files[i]);
    }
}
