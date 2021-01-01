# AI Edge Contest 4
This repository contains the source codes of the [4th AI Edge Contest on SIGNATE](https://signate.jp/competitions/285) of team Vertical-Beach.  
Implementation of semantic segmentation running on Ultra96-V2 using Vitis-AI and DPU.
The trained model is trained on a Japanese traffic image dataset, and the classification classes are road surface, traffic lights, pedestrians, cars, and others.  
The mIoU score on the contest evaluation data is 0.6014857.
[![](http://img.youtube.com/vi/0OF19EB_FHQ/0.jpg)](http://www.youtube.com/watch?v=0OF19EB_FHQ "")  

DNN model and inference code is based on [Xilinx ML-Caffe-Segmentation-Tutorial](https://github.com/Xilinx/Vitis-AI-Tutorials/tree/ML-Caffe-Segmentation-Tutorial).  
Training code is based on [fregu856/deeplabv3](https://github.com/fregu856/deeplabv3).  
Environments: Ubuntu18.04 and Vitis-AI v1.1

## Files
<!-- ファイル一覧についての説明 -->

## Demo  

## Training
### Model definition
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

We did not use model converter like ONNX because converted model cannot be quantized on `Vitis-AI Quantizer`. We manually extract model weight to numpy array and export it.

### Export pretrained caffe weight to numpy array
First, generate the model definition on pytorch by the lazy and ugly code: `util/model_convert_to_torch.py`  and extract pretrained weight of Xilinx tutorial. This step is performed on the Xilinx Vitis-AI docker environment.
```bash
conda activate vitis-ai-caffe
python model_convert_to_torch.py -i deploy.prototxt -o fpn_resnet18.py
python caffe_extract_to_numpy.py -i deploy.prototxt -w pretrained.caffemodel -o <numpy output dir>
``` 
### Export numpy array to pytorch and perform training
Next, setup pytorch training environment. We use [deepo](https://github.com/ufoym/deepo) for setup pytorch environment.  
You had better mount your Vitis-AI directory on deepo environment to share files among docker container.  
Export pretrained weight from numpy array to pytorch(.pth)
```bash
cd training
python ../util/export_from_numpy_to_torch.py -i <numpy output dir> -o ./pretrained.pth
```
You can modify `/training/datasets.py` for your custom dataset. 
```bash
cd training
python train.py
```
While training, the loss curve graph is updated for every epoch and generated to `/training/training_logs/<model_id>_loss.png` like:  
<!-- ここにlosscurveをはる -->

### Export pytorch weight to numpy and import to Caffe
After training finished, export pytorch weight to numpy array and import to caffe.  
```bash
#On pytorch envirionment
cd training
python ../util/export_pytorch_to_numpy.py -i epoch100.pth -o <numpy output dir>
#On Vitis-AI environment
python ../util/import_numpy_to_caffe.py -i <numpy output dir> -o converted.caffemodel
```

### Evaluate trained weight on host machine.


## Setup Vitis-AI platform on Ultra96-V2
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

## Software

### Evaluation for images
### Realtime Video app

## References
- https://signate.jp/competitions/28
- https://github.com/Xilinx/Vitis-AI-Tutorials/tree/ML-Caffe-Segmentation-Tutorial
- https://github.com/fregu856/deeplabv3
- https://github.com/bermanmaxim/LovaszSoftmax
## Licence
<!-- MIT? -->