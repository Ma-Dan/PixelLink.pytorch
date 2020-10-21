import os
import cv2
import sys
import time
import collections
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data

import models

import skimage.io as io
from skimage import transform

from torchsummary import summary
import torchvision.transforms as transforms

from scipy import ndimage as ndi

import matplotlib.pyplot as plt

def initModel(args):
    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=18, scale=args.scale)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=18, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=18, scale=args.scale)
    elif args.arch == "vgg16":
        model = models.vgg16(pretrained=True,num_classes=18)
    elif args.arch == "googlenet":
        model = models.googlenet_onnx(pretrained=True,num_classes=18)

    for param in model.parameters():
        param.requires_grad = False

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(("Loading model and optimizer from checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)

            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in list(checkpoint['state_dict'].items()):
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print(("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'])))
            sys.stdout.flush()
        else:
            print(("No checkpoint found at '{}'".format(args.resume)))
            sys.stdout.flush()

    model.eval()

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='vgg16')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')

    args = parser.parse_args()

    model = initModel(args)

    dummy_input = Variable(torch.randn(1, 3, 640, 640))
    torch.onnx.export(model, dummy_input, "pixellink.onnx", verbose=True, export_params=True, opset_version=11)