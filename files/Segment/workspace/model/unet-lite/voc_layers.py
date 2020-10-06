
import caffe

import numpy as np
from PIL import Image
import cv2

import random

class BDD100KDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        for SBDD semantic segmentation.
        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.
        example
        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.bdd100k_label_dir = params['bdd100k_label_dir']
        self.bdd100k_image_dir = params['bdd100k_image_dir']
        self.filelist = params['filelist']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.resize_size_y = int(params.get('resize_size_y', 256))
        self.resize_size_x = int(params.get('resize_size_x', 512))
        self.scale = params.get('scale', 0.022)
        self.batch_size = int(params.get('batch_size', 4))
        self.data_on_memory = params.get('data_on_memory', False)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.datapaths = open(self.filelist, 'r').read().splitlines()
        self.datas = []
        self.labels = []
        # self.datapaths = self.datapaths[:1]
        if self.data_on_memory:
            for i, datapath in enumerate(self.datapaths):
                print("loading " + str(i))
                self.datas.append(self.load_image(datapath))
                self.labels.append(self.load_label(datapath))

        random.seed(self.seed)

    def reshape(self, bottom, top):
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(self.batch_size, 3, self.resize_size_y, self.resize_size_x)
        top[1].reshape(self.batch_size, 1, self.resize_size_y, self.resize_size_x)


    def forward(self, bottom, top):
        # assign output
        for i in range(self.batch_size):
            idx = random.randint(0, len(self.datapaths)-1)
            if self.data_on_memory:
                top[0].data[i, ...] = self.datas[idx]
                top[1].data[i, ...] = self.labels[idx]
            else:
                top[0].data[i, ...] = self.load_image(self.datapaths[idx])
                top[1].data[i, ...] = self.load_label(self.datapaths[idx])

    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - multiply scale value
        - subtract mean
        - switch channels RGB -> BGR
        - transpose to channel x height x width order
        """
        im =  Image.open('{}/{}.jpg'.format(self.bdd100k_image_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = cv2.resize(in_, (self.resize_size_x, self.resize_size_y))
        in_ = in_ * self.scale
        in_ -= self.mean
        in_ = in_[:,:,::-1] 
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        pil_img = Image.open('{}/{}_train_id.png'.format(self.bdd100k_label_dir, idx))
        img = np.asarray(pil_img)
        #car 13 road 0 person 11 signal 6
        OTHER = 4
        img = np.where((img != 0) & (img != 6)  & (img != 11) & (img != 13), OTHER, img)
        #road 0 person 1 signal 2 car 3 other 4 
        img = np.where(img == 11, 1, img)
        img = np.where(img == 6, 2, img)
        img = np.where(img == 13, 3, img)
        img = cv2.resize(img, (self.resize_size_x, self.resize_size_y), interpolation=cv2.INTER_NEAREST)
        label = img[np.newaxis, ...]
        return label


class SignateDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        for SBDD semantic segmentation.
        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.
        example
        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.signate_label_dir = params['signate_label_dir']
        self.signate_image_dir = params['signate_image_dir']
        self.filelist = params['filelist']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.resize_size_y = int(params.get('resize_size_y', 256))
        self.resize_size_x = int(params.get('resize_size_x', 512))
        self.scale = params.get('scale', 0.022)
        self.batch_size = int(params.get('batch_size', 4))
        self.data_on_memory = params.get('data_on_memory', False)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.datapaths = open(self.filelist, 'r').read().splitlines()
        self.datas = []
        self.labels = []
        # self.datapaths = self.datapaths[:1]
        if self.data_on_memory:
            for i, datapath in enumerate(self.datapaths):
                print("loading " + str(i))
                self.datas.append(self.load_image(datapath))
                self.labels.append(self.load_label(datapath))

        random.seed(self.seed)

    def reshape(self, bottom, top):
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(self.batch_size, 3, self.resize_size_y, self.resize_size_x)
        top[1].reshape(self.batch_size, 1, self.resize_size_y, self.resize_size_x)


    def forward(self, bottom, top):
        # assign output
        for i in range(self.batch_size):
            idx = random.randint(0, len(self.datapaths)-1)
            if self.data_on_memory:
                top[0].data[i, ...] = self.datas[idx]
                top[1].data[i, ...] = self.labels[idx]
            else:
                top[0].data[i, ...] = self.load_image(self.datapaths[idx])
                top[1].data[i, ...] = self.load_label(self.datapaths[idx])

    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - multiply scale value
        - subtract mean
        - switch channels RGB -> BGR
        - transpose to channel x height x width order
        """
        im =  Image.open('{}/{}.jpg'.format(self.signate_image_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = cv2.resize(in_, (self.resize_size_x, self.resize_size_y))
        in_ = in_ * self.scale
        in_ -= self.mean
        in_ = in_[:,:,::-1] 
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        path = '{}/{}.dat'.format(self.signate_label_dir, idx)
        fp = open(path,'rb')
        dat = np.fromfile(fp, np.uint8, -1)
        fp.close()
        dat = dat.reshape((1216, 1936, 1))
        dat = cv2.resize(dat, (self.resize_size_x, self.resize_size_y), interpolation=cv2.INTER_NEAREST)
        #road 0 person 1 signal 2 car 3 other 4 
        label = dat[np.newaxis, ...]
        return label