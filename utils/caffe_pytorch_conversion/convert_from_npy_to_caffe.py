import caffe
import numpy as np
import argparse
import os

def converted_from_npy_to_caffe(caffe_model, npy_dir, out_path):
    #I don't know why, but use caffe.TRAIN, not caffe.TEST
    print(caffe_model)
    net = caffe.Net(caffe_model, caffe.TRAIN)
    for item in net.params.items():
        name, layer = item
        print(name)
        weight_num = len(net.params[name])
        name_replaced = name.replace('/', '_')
        load_num = -1
        reshape_flag = False
        if weight_num == 1:
            #conv or deconv without bias
            load_num = 1
        elif weight_num == 2:
            #conv or deconv with bias
            load_num = 2
        elif weight_num == 5:
            #batchnorm
            load_num = 4
            reshape_flag = True
        else:
            raise Exception('unknown weight num' + name)
        for i, p in zip(range(load_num), net.params[name]):
            path = os.path.join(npy_dir, name_replaced + '_' + str(i) + '.npy')
            if os.path.exists(path) is False:
                raise Exception(path)
            np_weight = np.load(path)
            if reshape_flag is True:
                np_weight = np.reshape(np_weight, (1, -1, 1, 1))
            assert(p.data.shape == np_weight.shape)
            # net.params[name].data = np_weight
            print(net.params[name][i].data.shape)
            net.params[name][i].data[...] = np_weight

    caffe.Net.save(net, out_path, caffe.TRAIN)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="caffe model prototxt")
    parser.add_argument("-i", "--npy_dir", help="numpy array weight file directory")
    parser.add_argument("-o", "--output", help="caffemodel output path")
    args = parser.parse_args()
    converted_from_npy_to_caffe(args.model, args.npy_dir, args.output)