import sys

from datasets import DatasetTrainSignate, DatasetValSignate # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from datasets import DatasetTrainBDD100K, DatasetValBDD100K
from utils.viz_graph import draw_loss_curve
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import cv2

# NOTE! NOTE! change this to not overwrite all log data when you train the model:
model_id = "bdd100k"

num_epochs = 2000
batch_size = 12
learning_rate = 0.007

from fpn_resnet18 import FPNResnet18
network = FPNResnet18(num_classes=5).cuda()
network.model_dir = "./training_logs/{}".format(model_id)
network.checkpoints_dir = "./training_logs/{}/checkpoints".format(model_id)

network.load_state_dict(torch.load("./pretrained.pth"))

import os
os.makedirs(network.model_dir, exist_ok=True)
os.makedirs(network.checkpoints_dir, exist_ok=True)

#choose dataloader
SIGNATE = 0
BDD100K = 1
DATASET_MODE = BDD100K
train_dataset = None
val_dataset = None
if DATASET_MODE == BDD100K:
    train_dataset = DatasetTrainBDD100K(bdd100k_data_path='../data/bdd100k')
    val_dataset = DatasetValBDD100K(bdd100k_data_path='../data/bdd100k')
elif DATASET_MODE == SIGNATE:
    train_dataset = DatasetTrainSignate(signate_data_path='../data/signate')
    val_dataset = DatasetValSignate(signate_data_path='../data/signate')
assert (train_dataset is not None and val_dataset is not None)
num_train_batches = int(len(train_dataset)/batch_size)
num_val_batches = int(len(val_dataset)/batch_size)
print ("num_train_batches:", num_train_batches)
print ("num_val_batches:", num_val_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=1)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=1)

optimizer = torch.optim.SGD(params=network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=4e-5)


loss_fn = None
SoftmaxCrossEntropy = 0
LovaszLoss = 1
#choose loss function
LOSS_MODE = LovaszLoss
if LOSS_MODE == SoftmaxCrossEntropy:
    class_weight_path = "./class_weights/bdd100k_class_weights.pkl" if DATASET_MODE == BDD100K else "./class_weights/signate_class_weights.pkl"
    with open(class_weight_path, "rb") as file:
        class_weights = np.array(pickle.load(file))
    class_weights = torch.from_numpy(class_weights)
    class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
elif LOSS_MODE == LovaszLoss:
    from lovasz_loss import LovaszSoftmax
    loss_fn = LovaszSoftmax()
assert(loss_fn is not None)

epoch_losses_train = []
epoch_losses_val = []

for epoch in range(num_epochs):
    print ("###########################")
    print ("######## NEW EPOCH ########")
    print ("###########################")
    print ("epoch: %d/%d" % (epoch+1, num_epochs))

    ############################################################################
    # train:
    ############################################################################
    network.train() # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs) in enumerate(train_loader):

        imgs = Variable(imgs).cuda()
        label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda()

        outputs = network(imgs)

        # compute the loss:
        loss = loss_fn(outputs, label_imgs)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        # optimization step:
        optimizer.zero_grad() # (reset gradients)
        loss.backward() # (compute gradients)
        optimizer.step() # (perform optimization step)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print ("train loss: %g" % epoch_loss)
    print ("####")

    ############################################################################
    # val:
    ############################################################################
    network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs, img_ids) in enumerate(val_loader):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(imgs).cuda()
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda()

            outputs = network(imgs) 

            # compute the loss:
            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print ("val loss: %g" % epoch_loss)
    loss_curve_path = os.path.join(network.model_dir, 'loss_curve.png')
    print(loss_curve_path)
    draw_loss_curve(epoch_losses_train, epoch_losses_val, loss_curve_path)
    # save the model weights to disk:
    # if epoch % 10 == 0:
    checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
    torch.save(network.state_dict(), checkpoint_path)
