'''
Original repository:
https://github.com/eriklindernoren/PyTorch-YOLOv3
modified from:
yolov3.cfg
'''
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .model_utils import build_targets, to_cpu, non_max_suppression, classify_targets
from .layers import Upsample, EmptyLayer, YOLOLayer, ADL


def conv_leaky(c_i: int, c_o: int, kernel_size=3, downsample=False):
    pad = (kernel_size - 1)//2
    if downsample:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=c_i,
                out_channels=c_o,
                kernel_size=kernel_size,
                stride=2,
                padding=pad
            ),
            nn.BatchNorm2d(c_o, momentum=0.9, eps=1e-5),
            nn.LeakyReLU(0.1)
        )
    return nn.Sequential(
        nn.Conv2d(
            in_channels=c_i,
            out_channels=c_o,
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        ),
        nn.BatchNorm2d(c_o, momentum=0.9, eps=1e-5),
        nn.LeakyReLU(0.1)
    )


def conv_twin(c_i: int):
    c_t = c_i//2
    return nn.Sequential(
        conv_leaky(c_i, c_t, 1),
        conv_leaky(c_t, c_i)
    )


def up_twin(c_i: int, c_o: int):
    return nn.Sequential(
        conv_leaky(c_i, c_o, 1),
        Upsample(scale_factor=2)
    )


class VehicleLayer(nn.Module):
    def __init__(self, c_i: int, c_t: int, vehicle_scale='small'):
        super(VehicleLayer, self).__init__()
        # 5 anchors per vehicle stage
        small_anchors = [(8, 12), (16, 22), (24, 24), (22, 16), (12, 8)]
        large_anchors = [(24, 36), (48, 66), (72, 72), (66, 48), (36, 24)]
        # # kmeans anchors
        # small_anchors = [(20, 35), (35, 21), (31, 32)]
        # large_anchors = [(26, 44), (44, 26), (75, 67)]

        if vehicle_scale not in ['large', 'small']:
            raise ValueError(
                'please select vehicle scale in {}'.format(['large', 'small']))
        if vehicle_scale == 'large':
            self.conv = nn.Sequential(
                conv_leaky(c_i, c_t),
                nn.Conv2d(
                    in_channels=c_t,
                    out_channels=5*len(large_anchors),
                    kernel_size=1,
                    stride=1
                ),
            )
            self.yolo = YOLOLayer(large_anchors)
        else:
            self.conv = nn.Sequential(
                conv_leaky(c_i, c_t),
                nn.Conv2d(
                    in_channels=c_t,
                    out_channels=5*len(small_anchors),
                    kernel_size=1,
                    stride=1
                ),
            )
            self.yolo = YOLOLayer(small_anchors)

    def forward(self, x, targets=None, img_dim=None, scene_conf=None):
        x = self.conv(x)
        box, loss = self.yolo(x, targets, img_dim, scene_conf)
        return box, loss


class DarknetStage(nn.Module):
    '''
    stage1 -> stage5 of darknet-53
    '''

    def __init__(self, stage_index: int):
        super(DarknetStage, self).__init__()
        twin_number = {
            1: 1,
            2: 2,
            3: 8,
            4: 8,
            5: 4
        }
        if stage_index not in twin_number:
            raise ValueError(
                'fail to init darknet-53 stage: error stage index')
        c_i = 2**(stage_index+4)
        c_o = 2*c_i
        self.conv = conv_leaky(c_i, c_o, 3, True)
        self.twins = nn.ModuleList()
        for _ in range(twin_number[stage_index]):
            self.twins.append(conv_twin(c_o))

    def forward(self, x):
        x = self.conv(x)
        for twin in self.twins:
            x = x + twin(x)
        return x


class DetectStage(nn.Module):
    '''
    detection module part
    '''

    def __init__(self, c_small: int = 256, c_large: int = 512):
        super(DetectStage, self).__init__()
        # large vehicle part
        cc = c_large + c_large//2
        self.up_large = up_twin(c_large, c_large//2)
        self.twin_large = nn.ModuleList()
        self.twin_large.append(
            nn.Sequential(
                conv_leaky(cc, c_large//2, 1),
                conv_leaky(c_large//2, c_large)
            )
        )
        self.twin_large.append(conv_twin(c_large))
        self.conv_large = conv_leaky(c_large, c_small)
        # conv -> C15 -> YoLo Layer
        self.large_vehicle = VehicleLayer(c_large//2, c_large, 'large')
        # small vehicle part
        self.up_small = up_twin(c_small, c_small//2)
        self.twin_small = nn.ModuleList()
        self.twin_small.append(
            nn.Sequential(
                conv_leaky(c_small+c_small//2, c_small//2, 1),
                conv_leaky(c_small//2, c_small)
            )
        )
        self.twin_small.append(conv_twin(c_small))
        self.conv_small = conv_leaky(c_small, c_small//2)
        # conv -> C15 -> YoLo Layer
        self.small_vehicle = VehicleLayer(c_small//2, c_small, 'small')
        # loss scale
        self.scale_large = 1.0
        self.scale_small = 1.0

    def forward(self, f_small, f_large, f_scene=None, targets=None, img_dim=None, scene_conf=None):
        # large vehicle part
        f_scene = self.up_large(f_scene)
        # concat in feature dim
        f_large = torch.cat([f_scene, f_large], 1)
        for twin in self.twin_large:
            f_large = twin(f_large)
        f_large = self.conv_large(f_large)
        large_boxes, loss_large = self.large_vehicle(
            f_large, targets, img_dim, scene_conf)
        # small vehicle part
        f_large = self.up_small(f_large)
        # concat in feature dim
        f_small = torch.cat([f_large, f_small], 1)
        for twin in self.twin_small:
            f_small = twin(f_small)
        f_small = self.conv_small(f_small)
        small_boxes, loss_small = self.small_vehicle(
            f_small, targets, img_dim, scene_conf)
        # print('large boxes: ',large_boxes.shape)
        # print('small boxes: ',small_boxes.shape)
        boxes = to_cpu(torch.cat([large_boxes, small_boxes], 1))
        loss = self.scale_large * loss_large + self.scale_small * loss_small
        return boxes if targets is None else (loss, boxes)



class SceneStage(nn.Module):
    '''
    scene classification
    '''

    def __init__(self, c_i: int, c_t: int = 15):
        super(SceneStage, self).__init__()
        self.conv = conv_leaky(c_i, c_i//2, 1)
        self.classify = nn.Sequential(
            conv_leaky(c_i//2, c_i),
            conv_leaky(c_i, c_t, 1),
            nn.Conv2d(
                in_channels=c_t,
                out_channels=1,
                kernel_size=1,
                stride=1
            )
        )

    def forward(self, x):
        f_scene = self.conv(x)
        x = self.classify(f_scene)
        x = torch.sigmoid(x)
        return x, f_scene


class SCAFNet(nn.Module):
    '''
    backbone: darknet 53
    scene context Attention-Based fusion Network
    '''

    def __init__(self, input_channels=3, img_size=512, adl_drop_rate=0.5, adl_drop_threshold=0.5):
        super(SCAFNet, self).__init__()
        self.img_size = img_size
        ############### backbone + #####################
        self.conv = conv_leaky(input_channels, 32)
        # downsample to 1/2
        self.stage1 = DarknetStage(1)
        # downsample to 1/4
        self.stage2 = DarknetStage(2)
        # downsample to 1/8
        self.stage3 = DarknetStage(3)
        # downsample to 1/16
        self.stage4 = DarknetStage(4)
        # downsample to 1/32
        self.stage5 = DarknetStage(5)
        ############### backbone - #####################
        ############### functional + #####################
        # detect module
        self.detect = DetectStage(256, 512)
        # scene module
        self.scene = SceneStage(1024)
        # adl module
        self.adl = ADL(adl_drop_rate=adl_drop_rate,
                       adl_drop_threshold=adl_drop_threshold)
        ############### functional - #####################

    def forward(self, x, targets=None):
        img_dim = (x.shape[2],x.shape[3])
        x = self.conv(x)
        # feature extract stages:
        x = self.stage1(x)
        x = self.stage2(x)
        # vehicle detection stages:
        f_small = self.stage3(x)
        f_large = self.stage4(f_small)
        # scene classify stage:
        x = self.adl(f_large)
        x = self.stage5(x)
        classify, f_scene = self.scene(x)
        if self.training:
            # joint training:
            # return scene loss + detection loss
            det_loss, boxes = self.detect(
                f_small, f_large, f_scene, targets, img_dim)
            # compute classify loss:
            cls_loss = classify_targets(classify, targets)
            loss = cls_loss + det_loss
            return loss, boxes
        else:
            boxes = self.detect(f_small, f_large, f_scene,
                                None, img_dim, scene_conf=classify)
            return boxes

###################################### below: pure detection part without scene module ########################### 

class PureDetect(nn.Module):
    '''
    detection module part
    detection without scene part 
    '''

    def __init__(self, c_small: int = 256, c_large: int = 512):
        super(PureDetect, self).__init__()
        # large vehicle part
        cc = c_large 
        # self.up_large = up_twin(c_large, c_large//2)
        self.twin_large = nn.ModuleList()
        self.twin_large.append(
            nn.Sequential(
                conv_leaky(cc, c_large//2, 1),
                conv_leaky(c_large//2, c_large)
            )
        )
        self.twin_large.append(conv_twin(c_large))
        self.conv_large = conv_leaky(c_large, c_small)
        # conv -> C15 -> YoLo Layer
        self.large_vehicle = VehicleLayer(c_large//2, c_large, 'large')
        # small vehicle part
        self.up_small = up_twin(c_small, c_small//2)
        self.twin_small = nn.ModuleList()
        self.twin_small.append(
            nn.Sequential(
                conv_leaky(c_small+c_small//2, c_small//2, 1),
                conv_leaky(c_small//2, c_small)
            )
        )
        self.twin_small.append(conv_twin(c_small))
        self.conv_small = conv_leaky(c_small, c_small//2)
        # conv -> C15 -> YoLo Layer
        self.small_vehicle = VehicleLayer(c_small//2, c_small, 'small')
        # loss scale
        self.scale_large = 1.0
        self.scale_small = 1.0

    def forward(self, f_small, f_large, f_scene=None, targets=None, img_dim=None, scene_conf=None):
        # # large vehicle part
        # f_scene = self.up_large(f_scene)
        # # concat in feature dim
        # f_large = torch.cat([f_scene, f_large], 1)
        for twin in self.twin_large:
            f_large = twin(f_large)
        f_large = self.conv_large(f_large)
        large_boxes, loss_large = self.large_vehicle(
            f_large, targets, img_dim, scene_conf)
        # small vehicle part
        f_large = self.up_small(f_large)
        # concat in feature dim
        f_small = torch.cat([f_large, f_small], 1)
        for twin in self.twin_small:
            f_small = twin(f_small)
        f_small = self.conv_small(f_small)
        small_boxes, loss_small = self.small_vehicle(
            f_small, targets, img_dim, scene_conf)
        # print('large boxes: ',large_boxes.shape)
        # print('small boxes: ',small_boxes.shape)
        boxes = to_cpu(torch.cat([large_boxes, small_boxes], 1))
        loss = self.scale_large * loss_large + self.scale_small * loss_small
        return boxes if targets is None else (loss, boxes)


class DetectNet(nn.Module):
    '''
    backbone: darknet 53
    without scene part
    '''

    def __init__(self, input_channels=3, img_size=512, adl_drop_rate=0.5, adl_drop_threshold=0.5):
        super(DetectNet, self).__init__()
        self.img_size = img_size
        ############### backbone + #####################
        self.conv = conv_leaky(input_channels, 32)
        # downsample to 1/2
        self.stage1 = DarknetStage(1)
        # downsample to 1/4
        self.stage2 = DarknetStage(2)
        # downsample to 1/8
        self.stage3 = DarknetStage(3)
        # downsample to 1/16
        self.stage4 = DarknetStage(4)
        # # downsample to 1/32
        # self.stage5 = DarknetStage(5)
        ############### backbone - #####################
        ############### functional + #####################
        # detect module
        self.detect = PureDetect(256, 512)
        # # scene module
        # self.scene = SceneStage(1024)
        # # adl module
        # self.adl = ADL(adl_drop_rate=adl_drop_rate,
        #                adl_drop_threshold=adl_drop_threshold)
        ############### functional - #####################

    def forward(self, x, targets=None):
        img_dim = (x.shape[2],x.shape[3])
        x = self.conv(x)
        # feature extract stages:
        x = self.stage1(x)
        x = self.stage2(x)
        # vehicle detection stages:
        f_small = self.stage3(x)
        f_large = self.stage4(f_small)
        # # scene classify stage:
        # x = self.adl(f_large)
        # x = self.stage5(x)
        # classify, f_scene = self.scene(x)
        f_scene = None 
        if self.training:
            # joint training:
            # return scene loss + detection loss
            det_loss, boxes = self.detect(
                f_small, f_large, f_scene, targets, img_dim)
            return det_loss, boxes
            # # compute classify loss:
            # cls_loss = classify_targets(classify, targets)
            # loss = cls_loss + det_loss
            # return loss, boxes
        else:
            boxes = self.detect(f_small, f_large, f_scene,
                                None, img_dim)
            return boxes
