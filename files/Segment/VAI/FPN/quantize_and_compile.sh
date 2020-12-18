#!/usr/bin/env bash
export GPUID=0
net=segmentation

#working directory
work_dir=$(pwd)
#path of float model
model_dir=quantize
#output directory
output_dir=compile

# echo "quantizing network: $(pwd)/float.prototxt"
# vai_q_caffe quantize          \
#           -model $(pwd)/float.prototxt     \
#           -weights $(pwd)/float.caffemodel \
#           -gpu $GPUID \
#           -calib_iter 1000 \
#           -output_dir ${model_dir} 2>&1 | tee ${model_dir}/quantize.txt

# echo "Compiling network: ${net}"

vai_c_caffe   --prototxt=${model_dir}/deploy.prototxt \
        --caffemodel=${model_dir}/deploy.caffemodel \
        --output_dir=${output_dir} \
        --net_name=${net} \
        --arch=/workspace/ultra96v2/ultra96v2_vitis_flow_tutorial_1/ULTRA96V2.json 2>&1 | tee ${output_dir}/compile.txt