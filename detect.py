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
    parser.add_argument("--det", )