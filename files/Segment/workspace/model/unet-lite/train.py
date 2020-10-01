import caffe
import sys
import os

import numpy as np
import sys

# weights = './final_models/pretrained.caffemodel'
# weights = './_iter_5000.caffemodel'
# weights = './bdd100k_trained/_iter_9000.caffemodel'
weights = './signate_trained/_iter_6000.caffemodel'
# init
# caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('./solver.prototxt')
solver.net.copy_from(weights)

# pretrained = caffe.Net('./deploy.prototxt', weights, caffe.TRAIN)
# for layer in pretrained.params:
#     if layer == 'conv_u0d-score_New':
#         continue
#     for i in range(0, len(pretrained.params[layer])): #this is for copying both weights and bias, in case bias exists
#         solver.net.params[layer][i].data[...]=np.copy(pretrained.params[layer][i].data[...])
#     print('Copy : ' + layer)


# scoring

# for _ in range(25):
solver.step(12000)