/**
* © Copyright (C) 2016-2017 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include <deque>
#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <sys/stat.h>
#include <sys/time.h>
#include <cstdio>
#include <iomanip>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// Header files for DNNDK APIs
#include <dnndk/dnndk.h>

using namespace std;
using namespace std::chrono;
using namespace cv;

// constant for segmentation network

#define KERNEL_CONV "segmentation_0"
#define CONV_INPUT_NODE "conv1_7x7_s2"
#define CONV_OUTPUT_NODE "toplayer_p2"
string seg_test_images = "../seg_test_images/";

// flags for each thread
bool is_reading = true;
bool is_running_1 = true;

queue<pair<string, string>> read_queue;     // read queue
mutex mtx_read_queue;     	 			 // mutex of read queue                                  


int dpuSetInputImageWithScale(DPUTask *task, const char* nodeName, const cv::Mat &image, float *mean, float scale, int idx)
{
    int value;
    int8_t *inputAddr;
    unsigned char *resized_data;
    cv::Mat newImage;
    float scaleFix;
    int height, width, channel;

    height = dpuGetInputTensorHeight(task, nodeName, idx);
    width = dpuGetInputTensorWidth(task, nodeName, idx);
    channel = dpuGetInputTensorChannel(task, nodeName, idx);

    if (height == image.rows && width == image.cols) {
        newImage = image;
    } else {
        newImage = cv::Mat (height, width, CV_8SC3,
                    (void*)dpuGetInputTensorAddress(task, nodeName, idx));
        cv::resize(image, newImage, newImage.size(), 0, 0, cv::INTER_LINEAR);
    }
    resized_data = newImage.data;

    inputAddr = dpuGetInputTensorAddress(task, nodeName, idx);
    scaleFix = dpuGetInputTensorScale(task, nodeName, idx);
    scaleFix = scaleFix*scale;

    if (newImage.channels() == 1) {
        for (int idx_h=0; idx_h<height; idx_h++) {
            for (int idx_w=0; idx_w<width; idx_w++) {
                for (int idx_c=0; idx_c<channel; idx_c++) {
                    value = *(resized_data+idx_h*width*channel+idx_w*channel+idx_c);
                    value = (int)((value - *(mean+idx_c)) * scaleFix);
                    inputAddr[idx_h*newImage.cols+idx_w] = (char)value;
                }
            }
        }
    } else {
        dpuProcessNormalizion(inputAddr, newImage.data, newImage.rows, newImage.cols, mean, scaleFix, newImage.step1());
    }
    return 0;
}


/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
vector<string> images;
void ListImages(string const &path, vector<string> &images) {
    images.clear();
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
                (ext == "PNG") || (ext == "png")) {
                images.push_back(name);
            }
        }
    }

    closedir(dir);
}

/**
 * @brief entry routine of segmentation, and put image into display queue
 *
 * @param task - pointer to Segmentation Task
 * @param is_running - status flag of the thread
 *
 * @return none
 */
void runSegmentation(DPUTask *task, bool &is_running) {
    // initialize the task's parameters
    DPUTensor *conv_out_tensor = dpuGetOutputTensor(task, CONV_OUTPUT_NODE);
    int outHeight = dpuGetTensorHeight(conv_out_tensor);
    int outWidth = dpuGetTensorWidth(conv_out_tensor);
    int8_t *outTensorAddr = dpuGetTensorAddress(conv_out_tensor);
    float mean[3]={73.0,82.0,72.0};
    float scale = 0.022;

    // Run detection for images in read queue
    while (is_running) {
        // Get an image from read queue
        Mat img;
        string filename;
        mtx_read_queue.lock();
        if (read_queue.empty()) {
            is_running = false;
            mtx_read_queue.unlock();
            break;
        } else {
            filename = read_queue.front().first;
            string img_path = read_queue.front().second;
            img = imread(img_path);
            read_queue.pop();
            mtx_read_queue.unlock();
        }

        std::chrono::system_clock::time_point  t1, t2, t3, t4, t5, t6;
        cout << filename << endl;
        t1 = std::chrono::system_clock::now();
        // Set image into CONV Task with mean value
		dpuSetInputImageWithScale(task, (char *)CONV_INPUT_NODE, img, mean, scale,0);							   
        t2 = std::chrono::system_clock::now();
        // Run CONV Task on DPU
        dpuRunTask(task);
        t3 = std::chrono::system_clock::now();
        Mat segMat(outHeight, outWidth, CV_8UC1);
        const int NUM_CLASS = 5;
        for (int row = 0; row < outHeight; row++) {
            for (int col = 0; col < outWidth; col++) {
                int i = row * outWidth * NUM_CLASS + col * NUM_CLASS;
                auto max_ind = max_element(outTensorAddr + i, outTensorAddr + i + NUM_CLASS);
                int posit = distance(outTensorAddr + i, max_ind);
                segMat.at<unsigned char>(row, col) = (unsigned char)(posit); //create a grayscale image with the class
            }
        }
        t4 = std::chrono::system_clock::now();
        resize(segMat, segMat, Size(1936,1216),0,0, INTER_NEAREST);
        t5 = std::chrono::system_clock::now();
        imwrite(filename,segMat); 
        t6 = std::chrono::system_clock::now();

        double preprocess =  (double)std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
        double dputask =     (double)std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();
        double postprocess = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count();
        double postresize  = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t5-t4).count();
        double savetime =    (double)std::chrono::duration_cast<std::chrono::milliseconds>(t6-t5).count();
        double alltime =     (double)std::chrono::duration_cast<std::chrono::milliseconds>(t6-t1).count();
        cout << "preprocess:" << preprocess << "[milisec]" << endl;
        cout << "dputask    :" << dputask << "[milisec]" << endl;
        cout << "postprocess:" << postprocess << "[milisec]" << endl;
        cout << "postresize :" << postresize << "[milisec]" << endl;
        cout << "savetime   :" << savetime << "[milisec]" << endl;
        cout << "alltime    :" << alltime << "[milisec]" << endl;
    }
}

/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
void Read(void) {
    // cout <<"...This routine assumes all images fit into DDR Memory..." << endl;
    cout <<"Reading Validation Images " << endl;
    ListImages(seg_test_images, images);
        if (images.size() == 0) {
            cerr << "\nError: No images exist in " << seg_test_images << endl;
            return;
        }   else {
            cout << "total images : " << images.size() << endl;
        }
        Mat img;
        string img_result_name;
        for (unsigned int img_idx=0; img_idx<images.size(); img_idx++) {
                cout << seg_test_images + images.at(img_idx) << endl;
                // img = imread(seg_test_images + images.at(img_idx));                                                     
                string img_path = seg_test_images + images.at(img_idx);                                                     
                img_result_name = "results/" + images.at(img_idx);
                img_result_name.erase(img_result_name.end()-4,img_result_name.end());
                img_result_name += ".png";
                read_queue.push(make_pair(img_result_name, img_path));
            }
    images.clear();
    cout << "...processing..." << endl;
}     
       

/**
 * @brief Entry for runing Segmentation neural network
 *
 * @arg file_name[string] - path to file for detection
 *
 */
int main(int argc, char **argv) {
    DPUKernel *kernel_conv;
    DPUTask *task_conv_1;

    // Attach to DPU driver and prepare for runing
    dpuOpen();
    // Create DPU Kernels and Tasks for CONV Nodes 
    kernel_conv = dpuLoadKernel(KERNEL_CONV);
    task_conv_1 = dpuCreateTask(kernel_conv, 0);
	
    Read();
    array<thread, 1> threads = {thread(runSegmentation, task_conv_1, ref(is_running_1))};

    for (int i = 0; i < 1; ++i) {
        threads[i].join();
    }
    cout << "evaluation completed, results stored in results folder" << endl;
    // Destroy DPU Tasks and Kernels and free resources
    dpuDestroyTask(task_conv_1);
    dpuDestroyKernel(kernel_conv);
    // Detach from DPU driver and release resources
    dpuClose();

    return 0;
}
