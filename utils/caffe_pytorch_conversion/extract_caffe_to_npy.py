import caffe
import numpy as np
import argparse
import os

def extract_caffe_model(model, weights, output_path):
  """extract caffe model's parameters to numpy array, and write them to files
  Args:
    model: path of '.prototxt'
    weights: path of '.caffemodel'
    output_path: output path of numpy params 
  Returns:
    None
  """
  net = caffe.Net(model, caffe.TEST)
  net.copy_from(weights)

  if not os.path.exists(output_path):
    os.makedirs(output_path)
  group_by_paramnum = {}
  for item in net.params.items():
    name, layer = item
    print('convert layer: ' + name)

    num = 0
    for p in net.params[name]:
      name = name.replace('/', '_')
      print(name, num, p.data.shape)
      np.save(output_path + '/' + str(name) + '_' + str(num), p.data)
      num += 1
    if num not in group_by_paramnum:
      group_by_paramnum[num] = []
    group_by_paramnum[num].append(name)
  
  for key, val in group_by_paramnum.items():
    print("---" + str(key) + "----")
    print('[{}]'.format(','.join(["'" + va + "'" for va in val])))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", help="model prototxt path .prototxt")
  parser.add_argument("-w", "--weights", help="caffe model weights path .caffemodel")
  parser.add_argument("-o", "--output", help="output path")
  args = parser.parse_args()
  extract_caffe_model(args.model, args.weights, args.output)