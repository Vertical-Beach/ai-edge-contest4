#include <iomanip>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

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
    glob((input_dir + "*.png").c_str(), 0, NULL, &globbuf);
    for (int i = 0; i < globbuf.gl_pathc; i++) {
        files.push_back(globbuf.gl_pathv[i]);
    }
    globfree(&globbuf);
    return files;
}

string get_basename(string path){
    string basename =  path.substr(path.find_last_of('/') + 1);
    basename = basename.substr(0, basename.size()-4);
    return basename;
}

void process(string path, bool flip){
    Mat img = imread(path);
    if(flip){
        cv::flip(img, img, 1);
    }
    uint8_t data[img.rows][img.cols];
    for(int y = 0; y < img.rows; y++){
        for(int x = 0; x < img.cols; x++){
            cv::Vec<unsigned char, 3> pix = img.ptr<cv::Vec3b>(y)[x];
            int b = pix[0];
            int g = pix[1];
            int r = pix[2];
            uint8_t index = 4;
            //road 0 person 1 signal 2 car 3 other 4
            if(r == 0 && g == 0 && b == 255) index = 3;
            if(r == 255 && g == 0 && b == 0) index = 1;
            if(r == 69 && g == 47 && b == 142) index = 0;
            if(r == 255 && g == 255 && b == 0) index = 2;
            data[y][x] = index;
        }
    }
    string basename = get_basename(path);
    string dstpath = "./seg_train_dat/" + basename + (flip ? "_flip" : "") + ".dat";
    FILE* fp = fopen(dstpath.c_str(), "wb");
    fwrite(data, sizeof(uint8_t), (int)img.cols*img.rows, fp);
    fclose(fp);
    cout << dstpath << endl;
    // cout << img.cols << " " << img.rows << endl;
}
int main(void){
    vector<string> files = get_file_path("../seg_train_annotations/");
    for(auto path: files){
        process(path, false);
        process(path, true);
    }
}