import caffe
import numpy as np

pretrained_weights = './final_models/pretrained.caffemodel'

caffe.set_mode_gpu()

solver = caffe.SGDSolver('./solver_bdd100k.prototxt')
pretrained = caffe.Net('./deploy_bdd100k.prototxt', pretrained_weights, caffe.TRAIN)
for layer in pretrained.params:
    if layer == 'toplayer_p2':
        continue
    # this is for copying both weights and bias, in case bias exists
    for i in range(0, len(pretrained.params[layer])):
        solver.net.params[layer][i].data[...] = np.copy(pretrained.params[layer][i].data[...])
    print('Copy : ' + layer)

solver.step(12000)
