"""
该文件通过解析yolov3.cfg文件来构建还原darknet
"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *


def parse_cfg(cfgfile):
    """
    解析darknet配置文件
    返回一个包含每个块的配置文件列表，每个块描述的是神经网络中的每一层的结构
    块的信息将会用字典的形式储存在列表中
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]  #去除空行
    lines = [x for x in lines if x[0] != '#'] #去除标注
    lines = [x.rstrip().lstrip() for x in lines]  #去除空格

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":    #每个块的开头
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)

    return blocks

#作用于route
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

#解析cfg文件
def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    #由于卷积核的维度是由上一层的卷积核数量决定，在这里用prev_filter来追踪卷积核的个数，初始化设为3表示RGB层
    prev_filters = 3
    output_filters = []

    #迭代模块列表，为每一个模块创建一个pytorch模块
    for index, x in enumerate(blocks[1:]):
        #初始化pytorch的序列模块00
        module = nn.Sequential()

        #检测每一个块的信息，通过检查块头信息，确定该块是哪一种层
        #每一个块的信息将会转换为pytorch的层信息

        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1)//2
            else:
                pad = 0

            #加入卷积网络
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            #添加批归一化层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            #检查激活函数
            if activation == "leaky":
                activn = nn.LeakyReLu(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)


        #如果是上采样层，我们使用bilinear
        #上采样层
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("Upsample_{}".format(index), upsample)


        #路由层
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #start of a route
            start = int(x["layer"][0])
            #end, if there exist one
            try:
                end = int(x["layers"][0])
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end = index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index),route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        #跳转连接层
        elif (x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        #YOLO 检测层
        elif (x["type"] == "yolo"):
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1] for i in range(0, len(anchors), 2))]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


#darknet
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info,  self,module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i


                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortchut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                #获取输入维度
                inp_dim = int(self.net_info["height"])

                #获取分类数量
                num_classes = int(module["classes"])

                #transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

                otuputs[i] = x

            return detections

        def load_weights(self, weighfile):
            fp = open(weighfile, "rb")

            #头5个值表示是一些头文件信息，包括
            #1.Major version number
            #2.Minor version number
            #3.Subversion number
            #4,5.Images seen by the network(during training)
            header = np.fromfile(fp, dtype = np.int32, count =5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]

            weights = np.fromfile(fp. dtype = np.float32)

            ptr = 0
            for i in range(len(self.module_list)):
                module_type = self.blocks[i + 1]["type"]

                #如果某块为卷积层，则加载权重参数，否则忽略
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    #获取批归一化层的参数个数
                    num_bn_biases = bn.bias.numel()

                    #加载参数
                    bn_biase = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weighs[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    #cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn.weights.view_as(bn.weights.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn.weights)
                    bn.running_mean.copt_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    #biases 数量
                    num_biases = conv.bias.numel()

                    #加载参数
                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                    ptr = ptr + num_biases

                    #将加载后的参数reshape成模型的维度
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #复制数据
                    conv.bias.data.copy_(conv_biases)

                #加载卷积层参数
                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weights.data.copy_(conv_weights)





