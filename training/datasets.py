import torch
import torch.utils.data

import numpy as np
import cv2
import os

def normalize_img(img):
    """
    normalize image (caffe model definition compatible)
    input: opencv numpy array image (h, w, c)
    output: dnn input array (c, h, w)
    """
    scale = 1.0
    mean = [104,117,123]
    img = img.astype(np.float32)
    img = img * scale
    img -= mean
    img = np.transpose(img, (2, 0, 1))
    return img

def transform_image_and_label(img, label, resize_h = 400, resize_w = 800, scale_min = 0.7, scale_max = 1.2, flip_flag = True, crop_size = 256):
    """
    transform image and label
    """
    #1. resize + random scaling
    scale = np.random.uniform(low=scale_min, high=scale_max)
    new_img_h = int(scale * resize_h)
    new_img_w = int(scale * resize_w)
    img = cv2.resize(img, (new_img_w, new_img_h), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (new_img_w, new_img_h), interpolation=cv2.INTER_NEAREST) 
    #2. random crop
    start_x = np.random.randint(low=0, high=(new_img_w - crop_size))
    end_x = start_x + crop_size
    start_y = np.random.randint(low=0, high=(new_img_h - crop_size))
    end_y = start_y + crop_size
    img = img[start_y:end_y, start_x:end_x]
    label = label[start_y:end_y, start_x:end_x]
    # flip
    flip = np.random.randint(low=0, high=2)
    if flip_flag and flip == 1:
        img = cv2.flip(img, 1)
        label = cv2.flip(label, 1)
    return img, label

class DatasetTrainSignate(torch.utils.data.Dataset):
    def __init__(self, signate_data_path):
        self.examples = []
        files = open(f"{signate_data_path}/train_basefiles.txt").read().split("\n")[:-1]
        for img_id in files:
            img_path = f"{signate_data_path}/seg_train_images/{img_id}.jpg"
            label_img_path = f"{signate_data_path}/seg_train_dat/{img_id}.dat"
            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path
            example["img_id"] = img_id
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1) 

        label_img_path = example["label_img_path"]
        fp = open(label_img_path,'rb')
        dat = np.fromfile(fp, np.uint8, -1)
        fp.close()
        label = dat.reshape((1216, 1936, 1))

        img, label = transform_image_and_label(img, label)
        img = normalize_img(img)

        # convert numpy -> torch:
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        return (img, label)

    def __len__(self):
        return self.num_examples

class DatasetValSignate(torch.utils.data.Dataset):
    def __init__(self, signate_data_path):
        self.examples = []
        files = open(f"{signate_data_path}/val_basefiles.txt").read().split("\n")[:-1]
        for img_id in files:
            img_path = f"{signate_data_path}/seg_train_images/{img_id}.jpg"
            label_img_path = f"{signate_data_path}/seg_train_dat/{img_id}.dat"
            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path
            example["img_id"] = img_id
            self.examples.append(example)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)

        label_img_path = example["label_img_path"]
        fp = open(label_img_path,'rb')
        dat = np.fromfile(fp, np.uint8, -1)
        fp.close()
        label = dat.reshape((1216, 1936, 1))

        img, label = transform_image_and_label(img, label)
        img = normalize_img(img)

        # convert numpy -> torch:
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        return (img, label, img_id)

    def __len__(self):
        return self.num_examples

class DatasetTestSignate(torch.utils.data.Dataset):
    def __init__(self, signate_data_path, new_img_h = 512, new_img_w = 1024):
        self.new_img_h = new_img_h
        self.new_img_w = new_img_w
        self.examples = []
        files = os.listdir(f'{signate_data_path}/seg_test_images/')
        for img_name in files:
            img_path = f"{signate_data_path}/seg_test_images/{img_name}"
            example = {}
            example["img_path"] = img_path
            example["img_id"] = img_name.replace('.jpg', '')
            self.examples.append(example)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1) 
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_LINEAR)

        img = normalize_img(img)

        # convert numpy -> torch:
        img = torch.from_numpy(img)

        return (img, img_id)

    def __len__(self):
        return self.num_examples
        
from PIL import Image
class DatasetTrainBDD100K(torch.utils.data.Dataset):
    def __init__(self, bdd100k_data_path):
        self.examples = []
        files = open(f"{bdd100k_data_path}/seg/train_files.txt").read().split("\n")[:-1]
        for img_id in files:
            img_path = f"{bdd100k_data_path}/seg/images/train/{img_id}.jpg"
            label_img_path = f"{bdd100k_data_path}/seg/labels/train/{img_id}_train_id.png"
            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path
            example["img_id"] = img_id
            self.examples.append(example)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)

        #BDD100k label image is pallet-mode image: pixel value means class value
        label_img_path = example["label_img_path"]
        pil_img = Image.open(label_img_path)
        label = np.asarray(pil_img)
        #we only use 5 class, so modify class id
        #car 13 road 0 person 11 signal 6 => road 0 person 1 signal 2 car 3 other 4
        OTHER = 4
        label = np.where((label != 0) & (label != 6)  & (label != 11) & (label != 13), OTHER, label)
        label = np.where(label == 11, 1, label)
        label = np.where(label == 6, 2, label)
        label = np.where(label == 13, 3, label)

        img, label = transform_image_and_label(img, label)
        img = normalize_img(img)

        # convert numpy -> torch:
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        return (img, label)

    def __len__(self):
        return self.num_examples

class DatasetValBDD100K(torch.utils.data.Dataset):
    def __init__(self, bdd100k_data_path,  new_img_h =None, new_img_w = None, transform_flag = True):
        self.transform_flag = transform_flag
        self.new_img_h = new_img_h
        self.new_img_w = new_img_w
        self.examples = []
        files = open(f"{bdd100k_data_path}/seg/val_files.txt").read().split("\n")[:-1]
        for img_id in files:
            img_path = f"{bdd100k_data_path}/seg/images/val/{img_id}.jpg"
            label_img_path = f"{bdd100k_data_path}/seg/labels/val/{img_id}_train_id.png"
            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path
            example["img_id"] = img_id
            self.examples.append(example)
        self.num_examples = len(self.examples)


    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)
        
        label_img_path = example["label_img_path"]
        pil_img = Image.open(label_img_path)
        label = np.asarray(pil_img)
        #we only use 5 class, so modify class id
        #car 13 road 0 person 11 signal 6 => road 0 person 1 signal 2 car 3 other 4
        OTHER = 4
        label = np.where((label != 0) & (label != 6)  & (label != 11) & (label != 13), OTHER, label)
        label = np.where(label == 11, 1, label)
        label = np.where(label == 6, 2, label)
        label = np.where(label == 13, 3, label)

        if self.transform_flag:
            img, label = transform_image_and_label(img, label)
        else:
            img = cv2.resize(img, (self.new_img_w, self.new_img_h), interpolation=cv2.INTER_LINEAR) 
            label = cv2.resize(label, (self.new_img_w, self.new_img_h), interpolation=cv2.INTER_NEAREST) 
        img = normalize_img(img)

        # convert numpy -> torch:
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        return (img, label, img_id)

    def __len__(self):
        return self.num_examples
        
class DatasetTestBDD100K(torch.utils.data.Dataset):
    def __init__(self, bdd100k_data_path, new_img_h = 512, new_img_w = 1024):
        self.new_img_h = new_img_h
        self.new_img_w = new_img_w
        self.examples = []
        files = open(f"{bdd100k_data_path}/seg/test_files.txt").read().split("\n")[:-1]
        for img_id in files:
            img_path = f"{bdd100k_data_path}/seg/images/test/{img_id}.jpg"
            example = {}
            example["img_path"] = img_path
            example["img_id"] = img_id
            self.examples.append(example)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1) 
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_LINEAR)

        img = normalize_img(img)

        # convert numpy -> torch:
        img = torch.from_numpy(img)

        return (img, img_id)

    def __len__(self):
        return self.num_examples