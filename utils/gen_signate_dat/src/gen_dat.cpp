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
    glob_files("jpg");
    glob_files("png");
    return files;
}

string get_basename(string path){
    string basename = path.substr(path.find_last_of('/') + 1);
    basename = basename.substr(0, basename.find_last_of('.'));
    return basename;
}

void process(const string& img_path,
             const string& anno_path,
             const bool flip) {
    Mat img  = imread(img_path);
    Mat anno = imread(anno_path);

    auto gen_dat = [](Mat& anno_img, Mat& out) {
        out = Mat::zeros(anno_img.size(), CV_8U);
        for(int y = 0; y < anno_img.rows; y++){
            for (int x = 0; x < anno_img.cols; x++) {
                cv::Vec<unsigned char, 3> pix = anno_img.ptr<cv::Vec3b>(y)[x];
                int b = pix[0];
                int g = pix[1];
                int r = pix[2];
                uint8_t index = 4;
                // road 0 person 1 signal 2 car 3 other 4
                if (r == 0 && g == 0 && b == 255)
                    index = 3;
                if (r == 255 && g == 0 && b == 0)
                    index = 1;
                if (r == 69 && g == 47 && b == 142)
                    index = 0;
                if (r == 255 && g == 255 && b == 0)
                    index = 2;
                out.at<uint8_t>(y, x) = index;
            }
        }
    };

    auto write_dat = [&](const Mat& out, const string& src_path, const string& basename_suffix = "") {
        const string dst_path = src_path.substr(0, src_path.find_last_of('/') + 1)
            + get_basename(src_path) + basename_suffix + ".dat";
        FILE* fp = fopen(dst_path.c_str(), "wb");
        fwrite(out.data, sizeof(uint8_t), (int)(out.cols * out.rows), fp);
        fclose(fp);
        cout << "generate " + dst_path << endl;
    };

    Mat out;
    gen_dat(anno, out);
    write_dat(out, anno_path);
    if(flip) {
        const auto dst_img_path = img_path.substr(0, img_path.find_last_of('/') + 1) + get_basename(img_path) + "_flip.jpg";
        Mat flip_img = img.clone();
        cv::flip(flip_img, flip_img, 1);
        imwrite(dst_img_path, flip_img);
        cout << "generate " + dst_img_path << endl;

        Mat flip_anno = anno.clone();
        cv::flip(flip_anno, flip_anno, 1);
        gen_dat(flip_anno, out);
        write_dat(out, anno_path, "_flip");
    }
}

int main(int argc, char *argv[]) {
    try {
        const bool flip = (1 < argc && string(argv[1]) == "flip") ? true : false;
        const auto img_env_path  = getenv("SIGNATE_TRAIN_IMG_DIR");
        const auto anno_env_path = getenv("SIGNATE_TRAIN_ANNO_DIR");
        const auto img_path  = (img_env_path != nullptr) ? img_env_path : "/workspace/Vitis-AI-Tutorials/files/Segment/workspace/data/signate/seg_train_images";
        const auto anno_path = (anno_env_path != nullptr) ? anno_env_path : "/workspace/Vitis-AI-Tutorials/files/Segment/workspace/data/signate/seg_train_annotations";

        std::cout << "the path of image data:      " << img_path << std::endl;
        std::cout << "the path of annotation data: " << anno_path << std::endl;

        vector<string> img_files = get_file_path(img_path);
        vector<string> anno_files = get_file_path(anno_path);
        if (img_files.size() != anno_files.size()) {
            throw std::logic_error("[FATAL ERROR] The number of image data and annotation data do not match");
        }

        for (size_t i = 0; i < img_files.size(); i++) {
            process(img_files[i], anno_files[i], flip);
        }
    } catch(const exception& e) {
        std::cout << e.what() << std::endl;
        exit(1);
    }
}
