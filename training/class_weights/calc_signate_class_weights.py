import numpy as np
import pickle
import cv2

num_classes = 5

trainId_to_count = {}
for trainId in range(num_classes):
    trainId_to_count[trainId] = 0

train_label_img_paths = []
trains = open("../../data/signate/train_basefiles.txt").read().split("\n")[:-1]
train_label_img_paths += trains
vals = open("../../data/signate/val_basefiles.txt").read().split("\n")[:-1]
train_label_img_paths += vals
print(len(train_label_img_paths))

# get the total number of pixels in all train label_imgs that are of each object class:
for step, label_img_path in enumerate(train_label_img_paths):
    print(step)
    
    path = "../../data/signate/seg_train_dat/" + label_img_path + ".dat"
    fp = open(path,'rb')
    dat = np.fromfile(fp, np.uint8, -1)
    fp.close()
    label_img = dat.reshape((1216, 1936, 1))

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

with open("./signate_class_weights.pkl", "wb") as file:
    pickle.dump(class_weights, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)