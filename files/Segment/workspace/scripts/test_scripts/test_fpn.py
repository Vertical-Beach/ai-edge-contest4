#!/usr/bin/env python
# © Copyright (C) 2016-2017 Xilinx, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may
# not use this file except in compliance with the License. A copy of the
# License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

#   Last Modified in 2021/01/03
#   by lp6m

import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
import caffe
import cv2
from PIL import Image

def label_img_to_color(img):
    #road 0 person 1 signal 2 car 3 other 4 
    label_to_color = {
        0: [142, 47, 69],
        1: [0, 0, 255],
        2: [0, 255, 255],
        3: [255, 0, 0],
        4: [0, 0, 0],
        }
    
    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])
    return img_color

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file')
    parser.add_argument('--input_image', type=str, required=True, help='input image path')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory in which the segmented images should be stored')
    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    caffe.set_mode_cpu()
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['pred'].data.shape
    input_image = cv2.imread(args.input_image, 1).astype(np.float32)
    input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
    b,g,r = cv2.split(input_image)
    h = input_image.shape[0]
    w = input_image.shape[1]
    scale = 1.0
    mean = [104, 117, 123]
    input_image = input_image * scale
    input_image -= mean
    input_image=cv2.merge((b,g,r))   
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.asarray([input_image])
    out = net.forward_all(**{net.inputs[0]: input_image})
    prediction = net.blobs['pred'].data[0].argmax(axis=0)
    prediction_rgb = label_img_to_color(prediction)
    orig = cv2.imread(args.input_image, 1)
    orig = cv2.resize(orig, (w, h))
    overlayed = prediction_rgb * 0.65 + orig * 0.35
    if args.out_dir is not None:
        input_path_ext = args.input_image.split(".")[-1]
        input_image_name = args.input_image.split("/")[-1:][0].replace('.' + input_path_ext, '')
        out_path_im = args.out_dir + input_image_name + '_fpn' + '.png'
        out_path_overlayed = args.out_dir + input_image_name + '_overlayed' + '.png'
        cv2.imwrite(out_path_im, prediction_rgb)
        cv2.imwrite(out_path_overlayed, overlayed)





