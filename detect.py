from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguments to the detect module
    """
    parser = argparse.ArgumentParser(description = "YOLO V3 Detection Module")

    parser.add_argument("--images", dest = "images", help =
                        "Image /Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help =
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help =
                        "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = "cfgfile", help = "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = "reso", help =
                        "Input resolution of the network. Increase to increase accuracy. Decrese to increase speed",
                        default = "416", type = str)

    return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()


num_classes = 80
classes = load_classes("data/coco.names")

#set up the neural network
print("Loading network .....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")


model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda()


#set the model in evaluation mode
model.eval()

read_dir = time.time()

#