'''
这个文件用于预处理数据集，使得其数据和标签可以直接用于训练
核心功能：
    读取所有的car,bus,truck标注，将其转化为json文件　
'''

import os 
import numpy as np 
import random 
import json  
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
import cv2 

def mkdir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)

def strId(x: int, l: int = 6):
    '''
    获取一个固定长度为 l 的id编号
    不足的前缀用0补齐
    '''
    pre = '0'*l
    s = pre+str(x)
    return s[-l:]


def sin(angle):
    return np.sin(angle*np.pi/180.0)


def cos(angle):
    return np.cos(angle*np.pi/180.0)


def normalBox(xcenter, ycenter, swidth, sheight, angle,style='center'):
    '''
    将一个旋转框，转换为包围其的最小正矩形框
    这个函数的正确性已经由 2020-09-24 早上的 verification.ipynb 验证通过
    fix bug: 2020-10-10早上10:20
    修改了一个系数，以让正矩形框更好的贴合在车辆边缘
    '''
    x = []
    y = []
    # 四个点的坐标
    # p1
    x.append(xcenter+swidth*cos(angle)+sheight*cos(angle-90.0))
    y.append(ycenter+swidth*sin(angle)+sheight*sin(angle-90.0))
    # p2
    x.append(xcenter+swidth*cos(angle)+sheight*cos(angle+90.0))
    y.append(ycenter+swidth*sin(angle)+sheight*sin(angle+90.0))
    # p3
    x.append(xcenter+swidth*cos(angle+180.0)+sheight*cos(angle-90.0))
    y.append(ycenter+swidth*sin(angle+180.0)+sheight*sin(angle-90.0))
    # p4
    x.append(xcenter+swidth*cos(angle+180.0)+sheight*cos(angle+90.0))
    y.append(ycenter+swidth*sin(angle+180.0)+sheight*sin(angle+90.0))
    xmin = (min(x))
    xmax = (max(x))
    ymin = (min(y))
    ymax = (max(y)) 
    def scale(angle):
        t = sin(2*angle)
        v = 0.3*abs(t)
        return 1.0-v
    s = scale(angle)
    xcenter = (xmin+xmax)/2 
    ycenter = (ymin+ymax)/2
    sheight = (ymax-ymin)/2
    swidth = (xmax-xmin)/2
    if style=='center':
        return int(xcenter),int(ycenter),int(2*s*swidth),int(2*s*sheight)
    # another box style
    xmin = int(xcenter - s*swidth)
    xmax = int(xcenter + s*swidth)
    ymin = int(ycenter - s*sheight)
    ymax = int(ycenter + s*sheight)
    return xmin, ymin, xmax, ymax



def getJson(dataPath: str='dlr'):
    '''
    save car, bus, truck boxes to boxes.json file
    '''
    imgs = []
    for name in os.listdir(dataPath):
        short, ext = os.path.splitext(name)
        if ext=='.JPG':
            imgs.append(short)
    # collect box labels
    labels = {}
    for img in imgs:
        labels[img] = []
        for suffix in ['_pkw.samp','_bus.samp','_truck.samp']:
            note_name = os.path.join(dataPath,img+suffix)
            # load labels 
            if not os.path.exists(note_name): 
                continue 
            with open(note_name,'r') as fp:
                lines = fp.readlines() 
            for line in lines:
                if line[0] in {'#','@'}:
                    continue 
                _, __, xcenter, ycenter, swidth, sheight, angle = [
                    float(s) for s in line.split()]
                box = normalBox(
                    xcenter, ycenter, swidth, sheight, angle)  
                labels[img].append(box) 
    json_name = 'boxes.json'
    with open(json_name,'w') as fp:
        json.dump(labels,fp) 

getJson()