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

/**
 *  Last Modified in 2020/12/25
 *  by medalotte, lp6m
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
#include <cstring>
#include <fstream>
#include <iostream>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <dnndk/dnndk.h>

#define DEBUG_MODE (0)

namespace {
    template<typename T>
    class ObjWithMtx {
    public:
        T obj;

        ObjWithMtx() = default;
        ObjWithMtx(const T& _obj) : obj(_obj) {}

        void operator =(const ObjWithMtx& obj) = delete;
        ObjWithMtx(const ObjWithMtx& obj) = delete;

        void lock() { mtx_.lock(); }
        bool try_lock() { return mtx_.try_lock(); }
        void unlock() { mtx_.unlock(); }

    private:
        std::mutex mtx_;
    };

    template<typename T, size_t D>
    class MultiThreadFIFO {
    public:
        explicit MultiThreadFIFO(const uint32_t& sleep_t_us = 100) :
            sleep_t_us_(sleep_t_us) {
            init();
        }

        void operator =(const MultiThreadFIFO& obj) = delete;
        MultiThreadFIFO(const MultiThreadFIFO& obj) = delete;

        void init() {
            std::unique_lock<std::mutex> lock_w_func(w_func_guard_, std::try_to_lock);
            std::unique_lock<std::mutex> lock_r_func(r_func_guard_, std::try_to_lock);
            if (!lock_w_func.owns_lock() || !lock_r_func.owns_lock()) {
                throw std::runtime_error("[ERROR] Initialization of the FIFO failed.");
            }
            for (auto& state : fifo_state_) {
                std::lock_guard<ObjWithMtx<ElementState>> lock_state(state);
                state.obj = ElementState::INVALID;
            }
            r_idx_ = 0;
            w_idx_ = 0;
        }

        void write(ObjWithMtx<bool>& no_abnormality, const bool& is_last, std::function<void(T&)> write_func) {
            std::unique_lock<std::mutex> lock_w_func(w_func_guard_, std::try_to_lock);
            if (!lock_w_func.owns_lock()) {
                throw std::runtime_error("[ERROR] The write function can't be called at the same time from multiple threads.");
            }
            while (true) {
                {
                    std::lock_guard<ObjWithMtx<ElementState>> lock_state(fifo_state_[w_idx_]);
                    if (fifo_state_[w_idx_].obj == ElementState::INVALID) {
                        break;
                    } else {
                        std::lock_guard<ObjWithMtx<bool>> lock(no_abnormality);
                        if (no_abnormality.obj == false) {
                            throw std::runtime_error("[ERROR] Terminate write process.");
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_t_us_));
            }
            {
                std::lock_guard<ObjWithMtx<T>> lock_fifo(fifo_[w_idx_]);
                write_func(fifo_[w_idx_].obj);
            }
            {
                std::lock_guard<ObjWithMtx<ElementState>> lock_state(fifo_state_[w_idx_]);
                fifo_state_[w_idx_].obj = is_last ? ElementState::VALID_LAST : ElementState::VALID;
            }
            incrementIdx(w_idx_);
        }

        void read(ObjWithMtx<bool>& no_abnormality, std::function<void(const T&)> read_func) {
            std::unique_lock<std::mutex> lock_r_func(r_func_guard_, std::try_to_lock);
            if (!lock_r_func.owns_lock()) {
                throw std::runtime_error("[ERROR] The read function can't be called at the same time from multiple threads.");
            }
            while (true) {
                {
                    std::lock_guard<ObjWithMtx<ElementState>> lock_state(fifo_state_[r_idx_]);
                    if (fifo_state_[r_idx_].obj == ElementState::VALID ||
                        fifo_state_[r_idx_].obj == ElementState::VALID_LAST) {
                        break;
                    } else {
                        std::lock_guard<ObjWithMtx<bool>> lock(no_abnormality);
                        if (no_abnormality.obj == false) {
                            throw std::runtime_error("[ERROR] Terminate read process.");
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_t_us_));
            }
            {
                std::lock_guard<ObjWithMtx<T>> lock_fifo(fifo_[r_idx_]);
                read_func(fifo_[r_idx_].obj);
            }
            {
                std::lock_guard<ObjWithMtx<ElementState>> lock_state(fifo_state_[r_idx_]);
                if (fifo_state_[r_idx_].obj == ElementState::VALID) {
                    fifo_state_[r_idx_].obj = ElementState::INVALID;
                    incrementIdx(r_idx_);
                } else {
                    fifo_state_[r_idx_].obj = ElementState::INVALID_LAST;
                }
            }
        }

        bool neverReadNextElement() {
            std::lock_guard<ObjWithMtx<ElementState>> lock_state(fifo_state_[r_idx_]);
            return (fifo_state_[r_idx_].obj == ElementState::INVALID_LAST);
        }

    private:
        enum class ElementState { VALID, VALID_LAST, INVALID, INVALID_LAST };

        void incrementIdx(size_t& idx) const {
            idx = (idx < D - 1) ? idx + 1 : 0;
        }

        const uint32_t sleep_t_us_;

        std::array<ObjWithMtx<T>, D> fifo_;
        std::array<ObjWithMtx<ElementState>, D> fifo_state_;
        std::mutex r_func_guard_, w_func_guard_;
        size_t r_idx_{0}, w_idx_{0};
    };

    struct DPUInOutInfo {
        cv::Size in_size, out_size;
        size_t in_channel, out_channel;
        int8_t* in_addr;
        int8_t* out_addr;
        float in_mean[3];
        float in_scale_fix;
    };

    std::ostream& operator << (std::ostream& os, const DPUInOutInfo& obj) {
        os << "[in_size : " << obj.in_size;
        os << ", out_size : " << obj.out_size;
        os << ", in_channel : " << obj.in_channel;
        os << ", in_scale_fix : " << obj.in_scale_fix;
        os << ", in_addr : " << static_cast<void*>(obj.in_addr);
        os << ", out_addr : " << static_cast<void*>(obj.out_addr);
        os << ", mean : [";
        for (size_t mi = 0; mi < 3; mi++) {
            os << obj.in_mean[mi] << ((mi != 2) ? ", " : "]]");
        }
        return os;
    };

    const std::string SEG_TEST_IMG_PATH = "../seg_test_images/";
    const std::string SEG_OUT_PATH      = "../seg_out/";
    const std::string KERNEL_CONV       = "segmentation_0";
    const std::string CONV_INPUT_NODE   = "conv1_7x7_s2";
    const std::string CONV_OUTPUT_NODE  = "toplayer_p2";

    constexpr auto BATCH_SIZE             = 30U;
    constexpr auto PREPROC_FIFO_DEPTH     = 3U;
    constexpr auto POSTPROC_FIFO_DEPTH    = 3U;
    constexpr auto SLEEP_T_US             = 100U;
    constexpr auto DPU_INPUT_IMG_WIDTH    = 960U;
    constexpr auto DPU_INPUT_IMG_HEIGHT   = 480U;
    constexpr auto DPU_INPUT_IMG_CHANNEL  = 3U;
    constexpr auto DPU_INPUT_IMG_SCALE    = 1.0f;
    constexpr auto DPU_INPUT_IMG_MEAN     = {104.0f, 117.0f, 123.0f};
    constexpr auto DPU_OUTPUT_IMG_WIDTH   = 960U;
    constexpr auto DPU_OUTPUT_IMG_HEIGHT  = 480U;
    constexpr auto DPU_OUTPUT_IMG_CHANNEL = 1U;
    constexpr auto DPU_OUTPUT_NOF_CLASS   = 5U;

    using PreprocFIFOElementType  = std::array<int8_t, DPU_INPUT_IMG_WIDTH * DPU_INPUT_IMG_HEIGHT * DPU_INPUT_IMG_CHANNEL>;
    using PostprocFIFOElementType = std::array<int8_t, DPU_OUTPUT_IMG_WIDTH * DPU_OUTPUT_IMG_HEIGHT * DPU_OUTPUT_IMG_CHANNEL>;

    ObjWithMtx<bool> no_abnormality(true);
    MultiThreadFIFO<PreprocFIFOElementType, PREPROC_FIFO_DEPTH> preproc_fifo(SLEEP_T_US);
    MultiThreadFIFO<PostprocFIFOElementType, POSTPROC_FIFO_DEPTH> postproc_fifo(SLEEP_T_US);

    DPUKernel* kernel_conv;
    DPUTask* task_conv_1;
    DPUTensor* conv_out_tensor;
    DPUInOutInfo dpu_inout_info;
    std::vector<cv::Mat> read_buffer, write_buffer;

#if DEBUG_MODE
    std::mutex cout_guard;
#endif

    void do_preprocess() {
        cv::Mat resized_img(dpu_inout_info.in_size, CV_8UC3);
        for (size_t im_i = 0; im_i < read_buffer.size(); im_i++) {
            /***** PREPROCESS FOR INFERENCE *****/
#if DEBUG_MODE
            const auto t0 = std::chrono::system_clock::now();
#endif

            cv::resize(read_buffer[im_i], resized_img, resized_img.size(), 0, 0, cv::INTER_LINEAR);

#if DEBUG_MODE
            const auto t1 = std::chrono::system_clock::now();
#endif

            const auto is_last = (bool)(im_i == read_buffer.size() - 1);
            preproc_fifo.write(no_abnormality, is_last, [&](PreprocFIFOElementType& dst) -> void {
                dpuProcessNormalizion(dst.data(), resized_img.data, resized_img.rows, resized_img.cols, dpu_inout_info.in_mean,
                                      dpu_inout_info.in_scale_fix, resized_img.step1());
            });

#if DEBUG_MODE
            const auto t2 = std::chrono::system_clock::now();
            const auto elapsed_time_ms0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)1e3;
            const auto elapsed_time_ms1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (double)1e3;
            {
                std::lock_guard<std::mutex> lock_cout(cout_guard);
                std::cout << "[DEBUG] (PREPROCESS) preproc_fifo.neverReadNextElement() : " << preproc_fifo.neverReadNextElement() << std::endl;
                std::cout << "[DEBUG] (PREPROCESS) postproc_fifo.neverReadNextElement() : " << postproc_fifo.neverReadNextElement() << std::endl;
                std::cout << "[DEBUG] (PREPROCESS) Elapsed time of cv::resize    : " << elapsed_time_ms0 << " ms" << std::endl;
                std::cout << "[DEBUG] (PREPROCESS) Elapsed time of normalization : " << elapsed_time_ms1 << " ms" << std::endl;
            }
#endif
            /***** END *****/
        }
    }

    void do_inference() {
        constexpr auto in_byte_size  = sizeof(int8_t) * DPU_INPUT_IMG_WIDTH * DPU_INPUT_IMG_HEIGHT * DPU_INPUT_IMG_CHANNEL;
        constexpr auto out_byte_size = sizeof(int8_t) * DPU_OUTPUT_IMG_WIDTH * DPU_OUTPUT_IMG_HEIGHT * DPU_OUTPUT_IMG_CHANNEL;

        auto end_flag = false;
        while (!end_flag) {
            /***** INFERENCE *****/
#if DEBUG_MODE
            const auto t0 = std::chrono::system_clock::now();
#endif

            preproc_fifo.read(no_abnormality, [&](const PreprocFIFOElementType& src) -> void {
                std::memcpy(dpu_inout_info.in_addr, src.data(), in_byte_size);
            });

#if DEBUG_MODE
            const auto t1 = std::chrono::system_clock::now();
#endif

            dpuRunTask(task_conv_1);

#if DEBUG_MODE
            const auto t2 = std::chrono::system_clock::now();
#endif

            end_flag = preproc_fifo.neverReadNextElement();
            postproc_fifo.write(no_abnormality, end_flag, [&](PostprocFIFOElementType& dst) -> void {
                std::memcpy(dst.data(), dpu_inout_info.out_addr, out_byte_size);
            });

#if DEBUG_MODE
            const auto t3 = std::chrono::system_clock::now();
            const auto elapsed_time_ms0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)1e3;
            const auto elapsed_time_ms1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (double)1e3;
            const auto elapsed_time_ms2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / (double)1e3;
            {
                std::lock_guard<std::mutex> lock_cout(cout_guard);
                std::cout << "[DEBUG] (INFERENCE) preproc_fifo.neverReadNextElement() : " << preproc_fifo.neverReadNextElement() << std::endl;
                std::cout << "[DEBUG] (INFERENCE) postproc_fifo.neverReadNextElement() : " << postproc_fifo.neverReadNextElement() << std::endl;
                std::cout << "[DEBUG] (INFERENCE) Elapsed time of memcpy from preproc FIFO : " << elapsed_time_ms0 << " ms" << std::endl;
                std::cout << "[DEBUG] (INFERENCE) Elapsed time of inference by DPU         : " << elapsed_time_ms1 << " ms" << std::endl;
                std::cout << "[DEBUG] (INFERENCE) Elapsed time of memcpy to postproce FIFO : " << elapsed_time_ms2 << " ms" << std::endl;
            }
#endif
            /***** END *****/
        }
    }

    void do_postprocess() {
        cv::Mat seg(dpu_inout_info.out_size, CV_8UC1);

        auto im_i = 0U;
        auto end_flag = false;
        while (!end_flag) {
            /***** POSTPROCESS OF INFERENCE *****/
#if DEBUG_MODE
            const auto t0 = std::chrono::system_clock::now();
#endif

            postproc_fifo.read(no_abnormality, [&](const PostprocFIFOElementType& dst) -> void {
                for (int ri = 0; ri < seg.rows; ri++) {
                    for (int ci = 0; ci < seg.cols; ci++) {
                        const auto idx = (ri * seg.cols + ci) * DPU_OUTPUT_NOF_CLASS;
                        const auto max_idx = std::max_element(dst.data() + idx, dst.data() + idx + DPU_OUTPUT_NOF_CLASS);
                        seg.at<unsigned char>(ri, ci) = (unsigned char)(std::distance(dst.data() + idx, max_idx));
                    }
                }
            });

#if DEBUG_MODE
            const auto t1 = std::chrono::system_clock::now();
#endif

            cv::resize(seg, write_buffer[im_i], write_buffer[im_i].size(), 0, 0, cv::INTER_NEAREST);

#if DEBUG_MODE
            const auto t2 = std::chrono::system_clock::now();
            const auto elapsed_time_ms0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)1e3;
            const auto elapsed_time_ms1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (double)1e3;
            {
                std::lock_guard<std::mutex> lock_cout(cout_guard);
                std::cout << "[DEBUG] (POSTPROCESS) preproc_fifo.neverReadNextElement() : " << preproc_fifo.neverReadNextElement() << std::endl;
                std::cout << "[DEBUG] (POSTPROCESS) postproc_fifo.neverReadNextElement() : " << postproc_fifo.neverReadNextElement() << std::endl;
                std::cout << "[DEBUG] (POSTPROCESS) Elapsed time of softmax    : " << elapsed_time_ms0 << " ms" << std::endl;
                std::cout << "[DEBUG] (POSTPROCESS) Elapsed time of cv::resize : " << elapsed_time_ms1 << " ms" << std::endl;
            }
#endif
            /***** END *****/

            end_flag = postproc_fifo.neverReadNextElement();
            im_i++;
        }
    }

    std::vector<std::string> load_img_names(const std::string& path) {
        const std::set<std::string> img_file_ext = {
            "JPEG", "jpeg", "JPG", "jpg", "PNG", "png"
        };

        struct stat s;
        lstat(path.c_str(), &s);
        if (!S_ISDIR(s.st_mode)) {
            throw std::runtime_error("Error: " + path + " is not a valid directory!\n");
        }

        DIR *dir = opendir(path.c_str());
        if (dir == nullptr) {
            throw std::runtime_error("Error: Open " + path + " path failed.\n");
        }

        struct dirent *entry;
        std::vector<std::string> img_names;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
                const std::string name = entry->d_name;
                const std::string ext  = name.substr(name.find_last_of(".") + 1);
                if (img_file_ext.find(ext) != img_file_ext.end()) {
                    img_names.push_back(name);
                }
            }
        }
        std::sort(img_names);

        closedir(dir);
        return img_names;
    }
}

int main() {
    try {
        std::exception_ptr ep;
        const auto gen_func = [&](const auto do_worker) -> auto {
            return [&]() -> void {
                try {
                    do_worker();
                }
                catch(...) {
                    std::lock_guard<ObjWithMtx<bool>> lock(no_abnormality);
                    if (no_abnormality.obj) {
                        no_abnormality.obj = false;
                        ep = std::current_exception();
                    }
                }
            };
        };

        // Attach to DPU driver and prepare for running
        dpuOpen();

        // Create DPU Kernels and Tasks for CONV Nodes
        kernel_conv     = dpuLoadKernel(KERNEL_CONV.c_str());
        task_conv_1     = dpuCreateTask(kernel_conv, 0);
        conv_out_tensor = dpuGetOutputTensor(task_conv_1, CONV_OUTPUT_NODE.c_str());

        dpu_inout_info.in_size.height  = dpuGetInputTensorHeight(task_conv_1, CONV_INPUT_NODE.c_str(), 0);
        dpu_inout_info.in_size.width   = dpuGetInputTensorWidth(task_conv_1, CONV_INPUT_NODE.c_str(), 0);
        dpu_inout_info.in_channel      = dpuGetInputTensorChannel(task_conv_1, CONV_INPUT_NODE.c_str(), 0);
        dpu_inout_info.out_size.height = dpuGetTensorHeight(conv_out_tensor);
        dpu_inout_info.out_size.width  = dpuGetTensorWidth(conv_out_tensor);

        const auto validate_dpu_inout = [](const bool& cond, const std::string& e_str) -> void {
            if (!cond) {
                throw std::logic_error("[ERROR] DPU in/out parameter is invalid : " + e_str);
            }
        };
        validate_dpu_inout(dpu_inout_info.in_size.width == DPU_INPUT_IMG_WIDTH, "input width");
        validate_dpu_inout(dpu_inout_info.in_size.height == DPU_INPUT_IMG_HEIGHT, "input height");
        validate_dpu_inout(dpu_inout_info.in_channel == DPU_INPUT_IMG_CHANNEL, "input channel");
        validate_dpu_inout(dpu_inout_info.out_size.width == DPU_OUTPUT_IMG_WIDTH, "output width");
        validate_dpu_inout(dpu_inout_info.out_size.height == DPU_OUTPUT_IMG_HEIGHT, "output height");

        dpu_inout_info.in_addr         = dpuGetInputTensorAddress(task_conv_1, CONV_INPUT_NODE.c_str(), 0);
        dpu_inout_info.in_scale_fix    = dpuGetInputTensorScale(task_conv_1, CONV_INPUT_NODE.c_str(), 0) * DPU_INPUT_IMG_SCALE;
        std::copy(DPU_INPUT_IMG_MEAN.begin(), DPU_INPUT_IMG_MEAN.end(), dpu_inout_info.in_mean);
        dpu_inout_info.out_addr        = dpuGetTensorAddress(conv_out_tensor);
        dpu_inout_info.out_channel     = DPU_OUTPUT_IMG_CHANNEL;

        std::cout << "[INFO] DPU information : " << dpu_inout_info << std::endl;

        const auto img_names = load_img_names(SEG_TEST_IMG_PATH);
        auto elapsed_time_ms = 0.0;
        for (size_t bi = 0; bi < std::ceil(img_names.size() / (double)BATCH_SIZE); bi++) {
            const auto idx_offset  = bi * BATCH_SIZE;
            const auto buffer_size = std::min((uint32_t)(img_names.size() - idx_offset), BATCH_SIZE);

            // Initialize buffers
            read_buffer.clear();
            write_buffer.clear();
            preproc_fifo.init();
            postproc_fifo.init();

            // Load images
            read_buffer.reserve(buffer_size);
            write_buffer.reserve(buffer_size);
            for (size_t im_i = 0; im_i < buffer_size; im_i++) {
                const std::string filepath = SEG_TEST_IMG_PATH + img_names[idx_offset + im_i];
                read_buffer.push_back(cv::imread(filepath));
                write_buffer.emplace_back(read_buffer.back().size(), CV_8UC1);
                std::cout << "[INFO] Load : " << filepath << std::endl;
            }

            std::cout << "[INFO] Perform inference for the " << idx_offset + 1 << "th to " << idx_offset + buffer_size << "th images." << std::endl;

            /***** START OF INFERENCE INCLUDING PRE/POST PROCESSES *****/
            const auto start_time = std::chrono::system_clock::now();

            auto preprocess  = std::thread(gen_func([&]() -> void { do_preprocess(); }));
            auto inference   = std::thread(gen_func([&]() -> void { do_inference(); }));
            auto postprocess = std::thread(gen_func([&]() -> void { do_postprocess(); }));

            preprocess.join();
            inference.join();
            postprocess.join();

            const auto end_time = std::chrono::system_clock::now();
            /***** END *****/

            if (ep) {
                std::rethrow_exception(ep);
            }

            if constexpr (!DEBUG_MODE) {
                elapsed_time_ms += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / (double)1e3;
                std::cout << "[INFO] Average elapsed time of inference for the 1th to "
                          << idx_offset + buffer_size + 1
                          << "th : "
                          << elapsed_time_ms / (idx_offset + buffer_size)
                          << " ms"
                          << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            for (size_t im_i = 0; im_i < buffer_size; im_i++) {
                std::string out_filepath = SEG_OUT_PATH + img_names[idx_offset + im_i];
                out_filepath  = out_filepath.substr(0, out_filepath.find_last_of("."));
                out_filepath += ".png";
                cv::imwrite(out_filepath, write_buffer[im_i]);
                std::cout << "[INFO] Store : " << out_filepath << std::endl;
            }
        }

        std::cout << "[INFO] Done." << std::endl;

        // Destroy DPU Tasks and Kernels and free resources
        dpuDestroyTask(task_conv_1);
        dpuDestroyKernel(kernel_conv);

        // Detach from DPU driver and release resources
        dpuClose();

    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    } catch (...) {
        std::cout << "An unexpected exception has occurred." << std::endl;
    }

    return 0;
}
