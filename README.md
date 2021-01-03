# AI Edge Contest 4
This repository contains the source codes of the [4th AI Edge Contest on SIGNATE](https://signate.jp/competitions/285) of team Vertical-Beach.  
Implementation of semantic segmentation running on Ultra96-V2 using Vitis-AI and DPU.
The trained model is trained on a Japanese traffic image dataset, and the classification classes are road surface, traffic lights, pedestrians, cars, and others. The training code also supports training on the BDD100K dataset.  
The mIoU score on the contest evaluation data is 0.6014857.  
Demo Video (Youtube Link):  
[![](http://img.youtube.com/vi/0OF19EB_FHQ/0.jpg)](http://www.youtube.com/watch?v=0OF19EB_FHQ "")  

DNN model and inference code is based on [Xilinx ML-Caffe-Segmentation-Tutorial](https://github.com/Xilinx/Vitis-AI-Tutorials/tree/ML-Caffe-Segmentation-Tutorial).  
Training code is based on [fregu856/deeplabv3](https://github.com/fregu856/deeplabv3).  
Environments: Ubuntu18.04 and Vitis-AI v1.1

# Files
- app  
  Evaluation applications on Ultra96
- data  
  put BDD100K/SIGNATE dataset here.
- dpu_info  
  DPU configuration data for Vitis-AI quantizer and compiler.
- files
  - Segment/workspace  
     model definition files for Xilinx Caffe.
  - Segment/VAI  
     model quantize and compile scripts for Vitis-AI quantizer and compiler.
- training  
    training scripts on PyTorch
- utils
  - weight convert scripts between pytorch and Caffe
  - signate dataset util

# Demo  
You can test segmentation app on Ultra96-V2 board.
Format SD image and divide SD into two partitions: first partition(512MB, FAT), second partition(extf4).
We labeled 'BOOT' and 'rootfs' for each partition.
```bash
/dev/sda1           2048  1050623  1048576   512M  b W95 FAT32
/dev/sda2        1050624 23490559 22439936  10.7G 83 Linux
```
```bash
git clone https://github.com/Vertical-Beach/ai-edge-contest4
cd prebuilt
#boot image
cp boot/* /media/<username>/rootfs/
#download petalinux rootfs.tar.gz from  https://drive.google.com/file/d/1CZCmH-_F6JzCOsH1YQWjfvzYKzBldcuu/view?usp=sharing
sudo tar xvf rootfs.tar.gz -C /media/<username>/rootfs/
#copy dpu image
cp dpu.xclbin /media/<username>/rootfs/usr/lib/
#download vitis-ai dnndk
wget -O vitis-ai_v1.1_dnndk.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.1_dnndk.tar.gz
tar -xvzf vitis-ai_v1.1_dnndk.tar.gz
sudo cp -r vitis-ai_v1.1_dnndk /media/<username>/home/root/
#copy realtime video application
sudo cp -r ../app/fpn_video /media/<username>/home/root/
```
After you created SD image, connect USB Mouse/Keyboard and DP-HDMI adapter, launch terminal on Ultra96-V2 board, and install Vitis-AI DNNDK.  
You can check your DPU configuration on board.
```bash
cd vitis-ai_v1.1_dnndk
source ./install.sh 
#run demo app
#see App section below for build application
cd ../fpn_video
sh change_resolution.sh
./segmentation video.mp4
```

# Training
## Model definition
We modified the `scale` and `mean` parameter of [FPN model of Xilinx tutorial](https://github.com/Xilinx/Vitis-AI-Tutorials/blob/ML-Caffe-Segmentation-Tutorial/files/Segment/workspace/model/FPN/train_val.prototxt#L7).
`mean` and `scale` was modified as following. These values are equal to the model definition of segmentation sample contained in Vitis-AI Model Zoo.  
```diff
<     mean_value: 73
<     mean_value: 82
<     mean_value: 72
<     scale: 0.022
---
>     mean_value: 104
>     mean_value: 117
>     mean_value: 123
>     scale: 1.0
```
This modification improves the inference accuracy on DPU for the following reason:
On the inference application using deployed DNN model on DPU, the image pixel value is normalized and inputted to the first layer based on the formula.
```
(pixval - mean) * scale * scaledFix
```
Here, `scaledFix` value is automatically calculated by Vitis-AI Quantizer, and compiled DPU model contains this value. You can get this value on `dpuGetInputTensorScale()` on DNNDK API.
`scaledFix` is a value to prevent the value calculated by the above formula from overflowing into the char range (-128 to 127). For example, when `scale=0.022`, calculated `scaledFix` value is 16, so image information of 256 gradations is reduced to 90 gradations (`256*0.022*16=90.1`). When `scale=1.0`, `scaledFix` value is 0.5 and pixel value is represented as 128 gradations, and it improves the inference accuracy.  
Furthermore, when `scale=0.861`, `scaledFix` will be 1.0 and pixel value will be represented as 256 gradations. (We did not do experiment.)

And also we modified the number of class of final output layer.
```diff
<     num_output: 19
---
>     num_output: 5
```

## Training on pytorch
On the referenced tutorial, the model training is performed on Xilinx Caffe framework and `SoftmaxWithCrossEntropy` loss functions is used. To improve the model accuracy, we used [Lovasz-Loss](https://github.com/bermanmaxim/LovaszSoftmax) function. The published `Lovasz-Loss` implementation is only for tensorflow and pytorch, and tensorflow implementation is very slow (see [issue](https://github.com/bermanmaxim/LovaszSoftmax/issues/6)). so we perform training on pytorch and export trained weight to caffe.  

We did not use model converter like ONNX because converted model cannot be quantized on `Vitis-AI Quantizer`. We manually extract model weight to numpy array and export it. On this step, we exlain training on BDD100K dataset.  

## Export pretrained caffe weight to numpy array
First, extract pretrained weight of Xilinx tutorial. This step is performed on the Xilinx Vitis-AI docker environment.
```bash
conda activate vitis-ai-caffe
cd files/Segment/workspace/model/FPN
python ../../../../../utils/caffe_pytorch_conversion/extract_caffe_to_npy.py -m deploy.prototxt -w final_models/pretrained.caffemodel  -o ../../../../../npy
``` 
Weights of each layer are generated in `npy` directory.  
## Export numpy array to pytorch and perform training
Next, setup pytorch training environment. We use [deepo](https://github.com/ufoym/deepo) for setup pytorch environment.  
You had better mount your Vitis-AI directory on deepo environment to share files between docker containers.
```bash
docker run --gpus all -v <path to Vitis-AI>/:/workspace -it  ufoym/deepo bin/bash
```
Export pretrained weight from numpy array to pytorch(.pth)  
The number of class is different from pretrained model(19), so this code doesn't copy weights of final layer.
```bash
cd training
python ../utils/caffe_pytorch_conversion/convert_from_npy_to_pytorch.py -i ../npy/ -o ./pretrained.pth
```
Download BDD100K datasets from here: https://bdd-data.berkeley.edu/  
Unzip bdd100k datasets and put `seg` directory to `data/bdd100k`.  
Modify `DATASET_MODE` and `LOSS_MODE` in `train.py`. and set `model_id` in `train.py`
```python
model_id = 'unique_name'
DATASET_MODE = BDD100K
LOSS_MODE = LovaszLoss
```
You can also train on your custom dataset by modifying `/training/datasets.py` and adding custom dataloader.
```bash
python train.py
```
While training, the loss curve graph is updated for every epoch and generated to `/training/training_logs/<model_id>/loss_curve.png` like:  
<!-- ここにlosscurveをはる -->
![Loss Curve](https://github.com/lp6m/aiedge_contest4/blob/prepare_publish/training/training_logs/bdd100k/loss_curve.png?raw=true "loss curve")

## Evaluate trained model on pytorch
set `DATASET_MODE=BDD100K` and `mode=TEST` in `training/evaluation/eval_on_val.py` for test.  
set `mode = VAL` for validation, and set `LOSS_MODE` for validation loss value function.  
set trained modedl path in `network.load_state_dict()`.
```bash
cd training/evaluation
python eval_on_val.py
```
label images are generated in `<model_id>/<test or val>/label`, and overlayed images are generated in `<model_id>/<test or val>/overlayed`.

## Export pytorch weight to numpy and import to Caffe
After training finished, export pytorch weight to numpy array and import to caffe.  
```bash
#On pytorch envirionment
cd training
python ../utils/caffe_pytorch_conversion/extract_pytorch_to_npy.py -i ./training_logs/bdd100k/checkpoints/model_bdd100k_epoch_334.pth -o ../npy2
```
Modify final output class in `files/Segment/workspace/model/FPN/deploy.prototxt`
```diff
<     num_output: 19
---
>     num_output: 5
```
```bash
#On Vitis-AI environment
conda activate vitis-ai-caffe
cd files/Segment/workspace/model/FPN/
python ../../../../../utils/caffe_pytorch_conversion/convert_from_npy_to_caffe.py -m ./deploy.prototxt -i ../../../../../npy2/ -o converted.caffemodel
```

## Evaluate trained weight on Xilinx caffe.
```bash
conda activate vitis-ai-caffe
cd files/Segment/workspace/scripts/test_scripts
sh test_fpn.sh
```
Output is slightly different between pytorch and caffe.
<!-- Caffeとpytorchの結果比較 -->
<!-- `val/overlayed/932a0bc8-753b2c12.png` -->

# Setup Vitis-AI platform on Ultra96-V2
English instruction is here : [Vitis-AI 1.1 Flow for Avnet VITIS Platforms - Part 1](https://www.hackster.io/AlbertaBeef/vitis-ai-1-1-flow-for-avnet-vitis-platforms-part-1-007b0e)  
Japanese instruction is here : [DNNDK on Vitis AI on Ultra96v2](https://qiita.com/nv-h/items/7525c9319087a3f51755#setup)  
In this project, we modified `<DPU-TRD>/prj/Vitis/config_file/prj_config` as follows.  
```
[clock]
id=4:dpu_xrt_top_1.aclk
id=5:dpu_xrt_top_1.ap_clk_2

[connectivity]

sp=dpu_xrt_top_1.M_AXI_GP0:HPC0
sp=dpu_xrt_top_1.M_AXI_HP0:HP0
sp=dpu_xrt_top_1.M_AXI_HP2:HP1


nk=dpu_xrt_top:1

[advanced]
misc=:solution_name=link
param=compiler.addOutputTypes=sd_card

#param=compiler.skipTimingCheckAndFrequencyScaling=1

[vivado]
prop=run.synth_1.strategy=Flow_AreaOptimized_high
prop=run.impl_1.strategy=Performance_ExtraTimingOpt
#param=place.runPartPlacer=0
```
`id=4, 5` means we give 200MHz and 400MHz for DPU IP.  We also modified  `[vivado]` option to change the vivado synthesis and implementation strategy and meet timing.

We disabled some layer support on DPU,  `dpu_conf.vh` was modified to reduce resource utilization.
```
`define POOL_AVG_DISABLE
`define DWCV_DISABLE
```
After following the tutorial and create an FPGA image using Vitis, the following files will be generated.
```bash
$DPU-TRD-ULTRA96V2/prj/Vitis/binary_container_1/sd_card ls
BOOT.BIN  README.txt  ULTRA96V2.hwh  dpu.xclbin  image.ub  rootfs.tar.gz
```
After you created SD image, connect USB Mouse/Keyboard and DP-HDMI adapter, and launch terminal on Ultra96-V2 board.
You can check your DPU configuration on board. 
```bash
$dexplorer -w
# ここにdexplorerの出力を貼る
```
I don't know why, but the clock frequency is displayed as 300MHz. But running time of DPU is faster than when implement on 150/300MHz, so we think displayed clock frequency on dexproler is not correct.  

## Quantize and Compile model
This step is same as the referenced tutorial. We add `-seg_class_num 5` for input arguments of `vai_q_caffe` for test quantized model. (Without this argument, the default number of classes 19 is used for measuring mIoU)
Modify `source_param` and `input_param` in `float.prototxt` for calibration process in Vitis-AI quantizer. (use absolute path).
Modify `arch` input arguments for `vai_c_caffe` in `quantize_and_compile.sh`. (use abso)
```bash
conda activate vitis-ai-caffe
cd files/Segment/VAI/FPN
cp <path-to converted.caffemodel> ./float.caffemodel
sh quantize_and_compile.sh
```
<details><summary>output example</summary>

```bash  
I1229 02:30:40.637688  6650 net.cpp:284] Network initialization done.
I1229 02:30:40.640810  6650 vai_q.cpp:182] Start Calibration
I1229 02:30:40.735327  6650 vai_q.cpp:206] Calibration iter: 1/1000 ,loss: 0
I1229 02:30:40.765679  6650 vai_q.cpp:206] Calibration iter: 2/1000 ,loss: 0

...
I1229 02:31:31.386523  6650 vai_q.cpp:360] Start Deploy
I1229 02:31:31.456115  6650 vai_q.cpp:368] Deploy Done!
--------------------------------------------------
Output Quantized Train&Test Model:   "quantize/quantize_train_test.prototxt"
Output Quantized Train&Test Weights: "quantize/quantize_train_test.caffemodel"
Output Deploy Weights: "quantize/deploy.caffemodel"
Output Deploy Model:   "quantize/deploy.prototxt"
...
Kernel topology "segmentation_kernel_graph.jpg" for network "segmentation"
kernel list info for network "segmentation"
                               Kernel ID : Name
                                       0 : segmentation_0
                                       1 : segmentation_1

                             Kernel Name : segmentation_0
--------------------------------------------------------------------------------
                             Kernel Type : DPUKernel
                               Code Size : 0.39MB
                              Param Size :
                Boundary Input Tensor(s)   (H*W*C)
                              score:0(0) : 320*640*5

               Boundary Output Tensor(s)   (H*W*C)
                              score:0(0) : 320*640*5

                           Input Node(s)   (H*W*C)
                                   score : 320*640*5

                          Output Node(s)   (H*W*C)
                                   score : 320*640*5
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
************************************************** 5.84MB
                           Workload MACs : 10934.67MOPS
                         IO Memory Space : 5.47MB
                              Mean Value : 104, 117, 123, 
                      Total Tensor Count : 129
                Boundary Input Tensor(s)   (H*W*C)
                               data:0(0) : 320*640*3

               Boundary Output Tensor(s)   (H*W*C)
                        toplayer_p2:0(0) :
                Boundary Input Tensor(s)   (H*W*C)
                              score:0(0) : 320*640*5

               Boundary Output Tensor(s)   (H*W*C)
                              score:0(0) : 320*640*5

                           Input Node(s)   (H*W*C)
                                   score : 320*640*5

                          Output Node(s)   (H*W*C)
                                   score : 320*640*5
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
************************************************** 95
                           Input Node(s)   (H*W*C)
                         conv1_7x7_s2(0) : 320*640*3

                          Output Node(s)   (H*W*C)
                          toplayer_p2(0) : 320*640*5




                             Kernel Name : segmentation_1
--------------------------------------------------------------------------------
                             Kernel Type : CPUKernel
                Boundary Input Tensor(s)   (H*W*C)
                              score:0(0) : 320*640*5

               Boundary Output Tensor(s)   (H*W*C)
                              score:0(0) : 320*640*5

                           Input Node(s)   (H*W*C)
                                   score : 320*640*5

                          Output Node(s)   (H*W*C)
                                   score : 320*640*5
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
```
</details>

# App
You can use cross-compile DPU application on host machine.  
First, setup cross compile environment. This step is peformed in host machine termianl.
```bash
cd ~/Downloads
wget -O sdk.sh https://www.xilinx.com/bin/public/openDownload?filename=sdk.sh
chmod +x sdk.sh
./sdk.sh -d ~/work/petalinux_sdk_vai_1_1_dnndk
unset LD_LIBRARY_PATH
source ~/work/petalinux_sdk_vai_1_1_dnndk/environment-setup-aarch64-xilinx-linux

wget -O vitis-ai_v1.1_dnndk.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.1_dnndk.tar.gz
tar -xvzf vitis-ai_v1.1_dnndk.tar.gz
cd vitis-ai_v1.1_dnndk
./install.sh $SDKTARGETSYSROOT
```
Copy compiled model to `model` directory before compile. You can use BDD100K/SIGNATE trained model.  
```bash
cp ./prebuilt/trained_model/bdd100k_segmentation.elf ./app/fpn_[video or eval]/model/dpu_segmentation_0.elf
cd app/fpn_[video or eval]
make
```

## Evaluation for images
Evaluation code for images is used for competition.  
There are three threads: preprocess, dpuprocess, postprocess.
Multi-thread processing reduced the average processing time per image.
<!-- PDF引用 -->
## Realtime Video app
Run realtime semantic segmentation for video. This application is based on `cf_fpn_cityscapes_256_512_8.9G` on [Xilinx's AI-Model-Zoo](https://github.com/Xilinx/AI-Model-Zoo). Input image size of DPU is modified to 256\*512 to 320\*640. Achieved FPS is about 15.5fps and it is slower than 27FPS described in AI-Model-Zoo.  
Demo Video (Youtube Link):  
[![](http://img.youtube.com/vi/0OF19EB_FHQ/0.jpg)](http://www.youtube.com/watch?v=0OF19EB_FHQ "")  
# References
- https://signate.jp/competitions/285
- https://github.com/Xilinx/Vitis-AI-Tutorials/tree/ML-Caffe-Segmentation-Tutorial
- https://github.com/fregu856/deeplabv3
- https://github.com/bermanmaxim/LovaszSoftmax
# Licence
<!-- MIT? -->