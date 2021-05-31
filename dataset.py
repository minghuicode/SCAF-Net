'''
Original repository:
https://github.com/eriklindernoren/PyTorch-YOLOv3
'''
import glob
import json
import random
import os
import sys
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size,
                          mode="nearest").squeeze(0)
    return image


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    if targets is None:
        return images, targets
    targets[:, 1] = 1 - targets[:, 1]
    return images, targets


class VehicleDataset(Dataset):
    def __init__(self, path, img_size=512, augment=True, multiscale=True):
        super(VehicleDataset, self).__init__()
        # 5 image for train, other 5 for test
        #            img name             box num
        # 2012-04-26-Muenchen-Tunnel_4K0G0070 590
        # 2012-04-26-Muenchen-Tunnel_4K0G0090 611
        # 2012-04-26-Muenchen-Tunnel_4K0G0020 337
        # 2012-04-26-Muenchen-Tunnel_4K0G0100 411
        # 2012-04-26-Muenchen-Tunnel_4K0G0060 725
        # 2012-04-26-Muenchen-Tunnel_4K0G0040 193
        # 2012-04-26-Muenchen-Tunnel_4K0G0080 277
        # 2012-04-26-Muenchen-Tunnel_4K0G0030 27
        # 2012-04-26-Muenchen-Tunnel_4K0G0051 247
        # 2012-04-26-Muenchen-Tunnel_4K0G0010 67
        self.path = path
        self.short = [
            '2012-04-26-Muenchen-Tunnel_4K0G0070',
            '2012-04-26-Muenchen-Tunnel_4K0G0090',
            '2012-04-26-Muenchen-Tunnel_4K0G0020',
            '2012-04-26-Muenchen-Tunnel_4K0G0100',
            '2012-04-26-Muenchen-Tunnel_4K0G0060'
        ]
        # check all images, if it exists?
        self.short = self.checkImages(path, self.short)
        self.img_size = img_size
        # max object per patch
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        # 416, 448, 480, 512, 544, 576, 608
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        with open('data/boxes.json', 'r') as fp:
            self.json = json.load(fp)
        # get first batch number
        self.batch_count = 0
        self.img, self.boxes = self.wholeImage()

    def checkImages(self, path, names):
        rv = []
        for t in names:
            img_name = os.path.join(path, t+'.JPG')
            if os.path.exists(img_name):
                rv.append(t)
        assert len(
            rv) > 0, 'found no data in train folder, please check dataset folder: {}'.format(path)
        return rv

    def wholeImage(self):
        t = random.choice(self.short)
        # store data to numpy with * B G R *
        image = cv2.imread(os.path.join(
            self.path, t+'.JPG'), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)
        _, self.height, self.width = img.shape
        boxes = self.json[t]
        return img, boxes

    def __getitem__(self, index):
        # ------
        # Image
        # ------
        xmin = random.randint(0, self.height-512)
        ymin = random.randint(0, self.width-512)
        xmax, ymax = xmin+512, ymin+512
        img = self.img[:, xmin:xmin+512, ymin:ymin+512]
        # ------
        #  Box
        # ------
        boxes = []
        # for cx, cy, h, w in self.boxes:
        for cy, cx, w, h in self.boxes:
            if xmin <= cx < xmax and ymin <= cy < ymax:
                x1, y1 = cx-h//2 - xmin, cy-w//2 - ymin
                x2, y2 = x1+h, y1+w
                x1, y1 = max(x1/512, 0), max(y1/512, 0)
                x2, y2 = min(x2/512, 1), min(y2/512, 1)
                # box = [(x1+x2)/2, (y1+y2)/2, (x2-x1), y2-y1]
                box = [(y1+y2)/2, (x1+x2)/2, (y2-y1), x2-x1]
                boxes.append(box)
        boxes = torch.from_numpy(np.array(boxes))
        if len(boxes) > 0:
            targets = torch.zeros((len(boxes), 5))
            targets[:, 1:] = boxes
        else:
            targets = None
        if self.augment:
            if random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        return img, targets

    def collate_fn(self, batch):
        imgs, targets = [], [] 
        # add sample index to targets
        for i, (img, boxes) in enumerate(batch):
            imgs.append(img)
            if boxes is None:
                continue
            boxes[:, 0] = i
            targets.append(boxes)
        if len(targets) > 0:
            targets = torch.cat(targets, 0)
        else:
            targets = None
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 4 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        if self.batch_count % 16 == 0:
            self.img, self.boxes = self.wholeImage() 
        return imgs, targets

    def __len__(self):
        # each epoch:
        #   # 5 images
        #   # each image:
        #   #   # 4 batch
        #       # each batch:
        #       #   # 16 imgs
        #   5x16 = 80 iters(20 optimizations) per epoch
        return len(self.short)*16*4 