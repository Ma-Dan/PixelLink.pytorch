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
        model = models.googlenet(pretrained=True,num_classes=18)

    for param in model.parameters():
        param.requires_grad = False

    #model = model.cuda()

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(("Loading model and optimizer from checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))

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

    summary(model, (3, 640, 640))

    return model

PIXEL_NEIGHBOUR_TYPE_4 = 'PIXEL_NEIGHBOUR_TYPE_4'
PIXEL_NEIGHBOUR_TYPE_8 = 'PIXEL_NEIGHBOUR_TYPE_8'

DECODE_METHOD_join = 'DECODE_METHOD_join'


def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
        (x - 1, y),                 (x + 1, y),  \
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def get_neighbours_4(x, y):
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]


def get_neighbours(x, y):
    import config
    neighbour_type = config.pixel_neighbour_type
    if neighbour_type == PIXEL_NEIGHBOUR_TYPE_4:
        return get_neighbours_4(x, y)
    else:
        return get_neighbours_8(x, y)

def get_neighbours_fn():
    import config
    neighbour_type = config.pixel_neighbour_type
    if neighbour_type == PIXEL_NEIGHBOUR_TYPE_4:
        return get_neighbours_4, 4
    else:
        return get_neighbours_8, 8

def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y >= 0 and y < h;

def decode_image_by_join(pixel_scores, link_scores, pixel_conf_threshold, link_conf_threshold):
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    done_mask = np.zeros(pixel_mask.shape, np.bool)
    result_mask = np.zeros(pixel_mask.shape, np.int32)
    points = list(zip(*np.where(pixel_mask)))
    h, w = np.shape(pixel_mask)
    group_id = 0
    for point in points:
        if done_mask[point]:
            continue
        group_id += 1
        group_q = [point]
        result_mask[point] = 1
        while len(group_q):
            y, x = group_q[-1]
            group_q.pop()
            if not done_mask[y,x]:
                done_mask[y,x], result_mask[y,x] = True, 1
                for n_idx, (nx, ny) in enumerate(get_neighbours(x, y)):
                    if is_valid_cord(nx, ny, w, h) and pixel_mask[ny, nx] and (link_mask[y, x, n_idx] or link_mask[ny, nx, 7 - n_idx]):
                        group_q.append((ny, nx))
    return result_mask

def decode_batch(pixel_cls_scores, pixel_link_scores,
                 pixel_conf_threshold = None, link_conf_threshold = None):
    import config
    if pixel_conf_threshold is None:
        pixel_conf_threshold = config.pixel_conf_threshold

    if link_conf_threshold is None:
        link_conf_threshold = config.link_conf_threshold

    batch_size = pixel_cls_scores.shape[0]
    batch_mask = []
    for image_idx in range(batch_size):
        image_pos_pixel_scores = pixel_cls_scores[image_idx, :, :]
        image_pos_link_scores = pixel_link_scores[image_idx, :, :, :]
        mask = decode_image(
            image_pos_pixel_scores, image_pos_link_scores,
            pixel_conf_threshold, link_conf_threshold
        )
        batch_mask.append(mask)
    return np.asarray(batch_mask, np.int32)

def decode_image(pixel_scores, link_scores,
                 pixel_conf_threshold, link_conf_threshold):
    import config
    if config.decode_method == DECODE_METHOD_join:
        mask =  decode_image_by_join(pixel_scores, link_scores,
                 pixel_conf_threshold, link_conf_threshold)
        return mask
    elif config.decode_method == DECODE_METHOD_border_split:
        return decode_image_by_border(pixel_scores, link_scores,
                 pixel_conf_threshold, link_conf_threshold)
    else:
        raise ValueError('Unknow decode method:%s'%(config.decode_method))

def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta].
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h

def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x

    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y

    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points

def get_shape(img):
    """
    return the height and width of an image
    """
    return np.shape(img)[0:2]

def resize(img, f = None, fx = None, fy = None, size = None, interpolation = cv2.INTER_LINEAR):
    """
    size: (w, h)
    """
    h, w = get_shape(img)
    if fx != None and fy != None:
        return cv2.resize(img, None, fx = fx, fy = fy, interpolation = interpolation)

    if size != None:
        return cv2.resize(img, size, interpolation = interpolation)

    return cv2.resize(img, None, fx = f, fy = f, interpolation = interpolation)

def find_contours(mask):
    mask = np.asarray(mask, dtype = np.uint8)
    mask = mask.copy()
    contours,_ = cv2.findContours(mask, mode = cv2.RETR_CCOMP,
                                   method = cv2.CHAIN_APPROX_SIMPLE)
    return contours

def mask_to_bboxes(mask, image_shape =  None, min_area = None,
                   min_height = None, min_aspect_ratio = None):
    import config
    feed_shape = config.train_image_shape

    if image_shape is None:
        image_shape = feed_shape

    image_shape = image_shape[:2]
    image_h, image_w = image_shape[:]

    if min_area is None:
        min_area = config.min_area

    if min_height is None:
        min_height = config.min_height
    bboxes = []
    mask = resize(img = mask, size = (image_w, image_h), interpolation = cv2.INTER_NEAREST)
    labeled_texts, count = ndi.label(mask)
    objects = ndi.find_objects(labeled_texts)
    for i in range(count):
        bbox_mask = np.zeros((image_h, image_w))
        bbox_mask[objects[i]] = mask[objects[i]]
        cnts = find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)

        w, h = rect[2:-1]
        if min(w, h) < min_height:
            continue

        if rect_area < min_area:
            continue

        # if max(w, h) * 1.0 / min(w, h) < 2:
        #     continue
        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)
    return bboxes

def to_bboxes(image_data, pixel_pos_scores, link_pos_scores):
    link_pos_scores=np.transpose(link_pos_scores,(0,2,3,1))
    mask = decode_batch(pixel_pos_scores, link_pos_scores,0.8,0.8)[0, ...]
    bboxes = mask_to_bboxes(mask, image_data.shape)
    return mask, bboxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='vgg16')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')

    args = parser.parse_args()

    model = initModel(args)

    while True:
        img = input('Input image filename:')
        try:
            org_img = io.imread(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            image = transform.resize(org_img, (640, 640)).astype('float32')
            text_box = org_img.copy()
            transform2 = transforms.Compose([
                transforms.ToTensor()
                #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
                ])
            image = torch.unsqueeze(transform2(image), 0) #.cuda()

            #torch.cuda.synchronize()
            cls_logits, link_logits = model(image)

            shape = link_logits.shape
            pixel_pos_scores = F.softmax(cls_logits, dim=1)[:,1,:,:]

            link_scores = link_logits.view(shape[0],2,8,shape[2],shape[3])
            link_pos_scores = F.softmax(link_scores, dim=1)[:,1,:,:,:]

            mask, bboxes = to_bboxes(org_img,pixel_pos_scores.numpy(), link_pos_scores.numpy())

            score = pixel_pos_scores[0,:,:]
            score = score.data.numpy().astype(np.float32)
            #torch.cuda.synchronize()

            for bbox in bboxes:
                cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)

            plt.imshow(text_box)
            plt.show()