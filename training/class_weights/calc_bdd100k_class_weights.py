import numpy as np
import pickle
import cv2
from PIL import Image
num_classes = 5

trainId_to_count = {}
for trainId in range(num_classes):
    trainId_to_count[trainId] = 0

train_label_img_paths = []
trains = open("../../data/bdd100k/seg/train_files.txt").read().split("\n")[:-1]
for train in trains:
    train_label_img_paths.append("../../data/bdd100k/seg/labels/train/" + train + "_train_id.png")
vals = open("../../data/bdd100k/seg/val_files.txt").read().split("\n")[:-1]
for val in vals:
    train_label_img_paths.append("../../data/bdd100k/seg/labels/val/" + val + "_train_id.png")
print(len(train_label_img_paths))

# get the total number of pixels in all train label_imgs that are of each object class:
for step, label_img_path in enumerate(train_label_img_paths):
    print(step)
    
    pil_img = Image.open(label_img_path)
    label_img = np.asarray(pil_img)
    #car 13 road 0 person 11 signal 6
    OTHER = 4
    label_img = np.where((label_img != 0) & (label_img != 6)  & (label_img != 11) & (label_img != 13), OTHER, label_img)
    #road 0 person 1 signal 2 car 3 other 4 
    label_img = np.where(label_img == 11, 1, label_img)
    label_img = np.where(label_img == 6, 2, label_img)
    label_img = np.where(label_img == 13, 3, label_img)

    for trainId in range(num_classes):
        # count how many pixels in label_img which are of object class trainId:
        trainId_mask = np.equal(label_img, trainId)
        trainId_count = np.sum(trainId_mask)

        # add to the total count:
        trainId_to_count[trainId] += trainId_count

# compute the class weights according to the ENet paper:
class_weights = []
total_count = sum(trainId_to_count.values())
for trainId, count in trainId_to_count.items():
    trainId_prob = float(count)/float(total_count)
    trainId_weight = 1/np.log(1.02 + trainId_prob)
    class_weights.append(trainId_weight)

print (class_weights)

with open("./bdd100k_class_weights.pkl", "wb") as file:
    pickle.dump(class_weights, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)