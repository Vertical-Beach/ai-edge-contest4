import sys
import os
sys.path.append(os.path.abspath(".."))
from datasets import DatasetValSignate, DatasetTestSignate # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from datasets import DatasetValBDD100K, DatasetTestBDD100K # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

from utils.utils import label_img_to_color_signate

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import cv2

batch_size = 1
model_id = "bdd100k"
from fpn_resnet18 import FPNResnet18
network = FPNResnet18().cuda()
# network.load_state_dict(torch.load(f"../training_logs/{model_id}/checkpoints/model_signate_epoch_1262.pth"))
network.load_state_dict(torch.load(f"../training_logs/{model_id}/checkpoints/model_bdd100k_epoch_33.pth"))

#choose dataset
SIGNATE = 0
BDD100K = 1
DATASET_MODE = BDD100K
#choose validation or test
VAL = 0
TEST = 1
mode = VAL
dataset = None
if mode == VAL:
    if DATASET_MODE == BDD100K:
        dataset = DatasetValBDD100K(bdd100k_data_path='../../data/bdd100k')
    elif DATASET_MODE == SIGNATE:
        dataset = DatasetValSignate(signate_data_path='../../data/signate')
else:
    if DATASET_MODE == BDD100K:
        dataset = DatasetTestBDD100K(bdd100k_data_path='../../data/bdd100k', new_img_h = 320, new_img_w = 640)
    elif DATASET_MODE == SIGNATE:
        dataset = DatasetTestSignate(signate_data_path='../../data/signate', new_img_h = 320, new_img_w = 640)
dirname = 'val' if mode == VAL else 'test'
os.makedirs(f'./{model_id}/{dirname}', exist_ok=True)
os.makedirs(f'./{model_id}/{dirname}/label', exist_ok=True)
os.makedirs(f'./{model_id}/{dirname}/overlayed', exist_ok=True)

num_val_batches = int(len(dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

data_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size, shuffle=False,num_workers=1)

loss_fn = None
SoftmaxCrossEntropy = 0
LovaszLoss = 1
#choose loss function
LOSS_MODE = LovaszLoss
if LOSS_MODE == SoftmaxCrossEntropy:
    class_weight_path = "../class_weights/bdd100k_class_weights.pkl" if DATASET_MODE == BDD100K else "../class_weights/signate_class_weights.pkl"
    with open(class_weight_path, "rb") as file:
        class_weights = np.array(pickle.load(file))
    class_weights = torch.from_numpy(class_weights)
    class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
elif LOSS_MODE == LovaszLoss:
    from lovasz_loss import LovaszSoftmax
    loss_fn = LovaszSoftmax()
assert(loss_fn is not None)

network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
batch_losses = []
import pickle;
for step, data in enumerate(data_loader):
    if mode == VAL:
        imgs, label_imgs, img_ids = data
    else:
        imgs, img_ids = data
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))
        if mode == VAL:
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))
            # compute the loss:
            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)
            print('loss', loss_value)

        ########################################################################
        # save data for visualization:
        ########################################################################
        outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
        # with open('./fpn_resnet18_continue/pkl/' + img_ids[0] + '.pkl', 'wb') as f:
            # pickle.dump(outputs, f)
        pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)

        for i in range(pred_label_imgs.shape[0]):
            if i == 0:
                pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))
                img_id = img_ids[i]
                img = imgs[i] # (shape: (3, img_h, img_w))

                img = img.data.cpu().numpy()
                #denormalize
                img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
                img = img + np.array([104, 117, 123])
                img = img / 1.0
                img = img.astype(np.uint8)

                pred_label_img_color = label_img_to_color_signate(pred_label_img)
                overlayed_img = 0.35*img + 0.65*pred_label_img_color
                overlayed_img = overlayed_img.astype(np.uint8)
                
                dst_overlayed = f"./{model_id}/{dirname}/overlayed/{img_id}.png"
                dst_labeled =   f"./{model_id}/{dirname}/label/{img_id}.png"
                print(dst_labeled, dst_overlayed)
                cv2.imwrite(dst_overlayed, overlayed_img)
                cv2.imwrite(dst_labeled, pred_label_img_color)

if mode == VAL:
    val_loss = np.mean(batch_losses)
    print ("val loss: %g" % val_loss)
