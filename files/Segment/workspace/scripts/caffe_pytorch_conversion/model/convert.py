#unko
lines = open('fpn_tf.py').read().split('\n')

def str_to_dic(argstrs):
    dic = {}
    for argstr in argstrs:
        key = argstr[:argstr.rindex('=')].strip()
        value = argstr[argstr.rindex('=')+1:].strip()
        if key == 'name':
            value = value[1:-1]
        dic[key] = value
    return dic

out_layer_channel_dic = {'inputs':'3'}
layer_lines = []
forward_lines = []
for line in lines:
    if len(line.strip()) == 0:
        continue
    line = line.strip()
    args = line[line.index('(')+1: line.rindex(')')].split(',')
    if 'FPN.conv' in line:
        input = args[0]
        c_i = out_layer_channel_dic[input]
        k_size = args[1]
        c_o = args[3]
        stride = args[4]
        dic = str_to_dic(args[6:])
        name = dic['name']
        out_layer_channel_dic[name] = c_o
        #assue always relu False
        if dic['relu'] != 'False':
            raise Exception()
        #assume always put bias
        if 'biased' in dic and dic['biased'] != 'True':
            raise Exception()
        k_size = int(k_size.strip())
        assert(k_size % 2 == 1)
        layer_line = f'self.l_{name} = nn.Conv2d({c_i}, {c_o}, {k_size}, {stride}, bias=True, padding={k_size//2})'
        forward_line = f'{name} = self.l_{name}({input})'
        layer_lines.append(layer_line)
        forward_lines.append(forward_line)
    elif 'FPN.deconv' in line:
        input = args[0]
        c_i = out_layer_channel_dic[input]
        k_size = args[1]
        c_o = args[3]
        out_layer_channel_dic[name] = c_o
        stride = args[4]
        dic = str_to_dic(args[6:])
        padding = dic['padding']
        name = dic['name']
        #some layers does not have bias
        if 'biased' not in dic:
            biased = True
        else:
            biased = (dic['biased'] == 'True')
        layer_line = f'self.l_{name} = nn.ConvTranspose2d({c_i}, {c_o}, {k_size}, {stride}, bias={str(biased)}, padding={padding})'
        forward_line = f'{name} = self.l_{name}({input})'
        layer_lines.append(layer_line)
        forward_lines.append(forward_line)
    elif 'FPN.batch_normalization' in line:
        #always relu=True
        input = args[0]
        dic = str_to_dic(args[1:])
        name = dic['name']
        out_layer_channel_dic[name] = out_layer_channel_dic[input]
        c_o = out_layer_channel_dic[name]
        layer_line = f'self.l_{name} = nn.BatchNorm2d({c_o})'
        forward_line = f'{name} = relu(self.l_{name}({input}))'
        layer_lines.append(layer_line)
        forward_lines.append(forward_line)
    elif 'FPN.add' in line:
        addlist = line[line.index('[')+1:line.index(']')].split(',')
        args = line[line.index(']')+2:line.rindex(')')].strip().split(',')
        dic = str_to_dic(args)
        name = dic['name']
        c_o = 32
        out_layer_channel_dic[name] = c_o
        s = ','.join(addlist)
        forward_line = f'{name} = torch.add({s})'
        forward_lines.append(forward_line)

    elif 'FPN.concat' in line:
        #concat axis always 1
        addlist = line[line.index('[')+1:line.index(']')].split(',')
        args = line[line.index(']')+5:line.rindex(')')].strip().split(',')
        dic = str_to_dic(args)
        name = dic['name']
        c_o = -1
        if name == 'inception_3a_output':
            c_o = 256
        elif name == 'inception_3b_output':
            c_o = 480
        elif name == 'inception_4a_output':
            c_o = 512
        elif name == 'inception_4b_output':
            c_o = 512
        elif name == 'inception_4c_output':
            c_o = 512
        elif name == 'inception_4d_output':
            c_o = 528
        elif name == 'inception_4e_output':
            c_o = 832
        elif name == 'inception_5a_output':
            c_o = 832
        elif name == 'inception_5b_output':
            c_o = 1024
        else:
            raise Exception()
        out_layer_channel_dic[name] = c_o
        s = ','.join(addlist)
        forward_line = f'{name} = torch.cat(({s}), dim=1)'
        forward_lines.append(forward_line)

    elif 'FPN.max_pool' in line:
        input = args[0]
        k_size = args[1]
        stride = args[3]
        dic = str_to_dic(args[5:])
        name = dic['name']
        out_layer_channel_dic[name] = out_layer_channel_dic[input]
        k_size = int(k_size.strip())
        assert(k_size % 2 == 1)
        layer_line = f'self.l_{name} = nn.MaxPool2d({k_size}, {stride}, padding={(k_size//2)})'
        forward_line = f'{name} = self.l_{name}({input})'
        layer_lines.append(layer_line)
        forward_lines.append(forward_line)
    else:
        raise Exception('unknown type layer : ' + line)

print('----init---')
for layer_line in layer_lines:
    print(layer_line)

print('----forward---')
for forward_line in forward_lines:
    print(forward_line)
