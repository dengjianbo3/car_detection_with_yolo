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


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #检测每一个块的信息，通过检查块头信息，确定该块是哪一种层

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

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("Upsample_{}".format(index), upsample)


        #路由层
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')

            start = int(x["layer"][0])