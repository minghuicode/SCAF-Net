'''
read all images from input folder
and save the detection result to output folder
'''

import os
import sys
import time
import datetime
import argparse
import tqdm
import json
import cv2

from models import darknet
from models.model_utils import nmsTest, xyxy2xywh
from mAP.mAP_utils import ap_iou
import time
import numpy as np
import json 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

device = 'cuda' 


if __name__ == "__main__":
    '''
    parse configs and load files
    '''
    # parse configs 
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',"--conf_thres", type=float, default=0.7,
                        help="confidence threshold")
    parser.add_argument('-n',"--nms_thres", type=float, default=0.5,
                        help="non max suppress threshold") 
    parser.add_argument('-g',"--gsd", type=float, default=12.5,
                        help="ground sample distance of aerial image, default 12.5cm/pixel")
    parser.add_argument('-o',"--overlap", type=int, default=32,
                        help="overlap of focus cover method")
    parser.add_argument('-s',"--full_size", type=int, default=512,
                        help="overlap of focus cover method")
    parser.add_argument('-f',"--file", type=str, default='checkpoints/scaf.pth',
                        help="model weights file")    
    parser.add_argument("--img_folder", type=str, default='input',
                        help="source image folder")   
    parser.add_argument("--result_folder", type=str, default='output',
                        help="output image folder") 
    parser.add_argument("--no_scene", action="store_true",
                        help="network without scene branch")   
    parser.add_argument("--evaluation", action="store_true",
                        help="not predict but evaluation")
    opt = parser.parse_args()
    print(opt) 
    assert opt.gsd>0
    ### load models
    t1 = time.time()
    if opt.no_scene:
        model = darknet.DetectNet().to(device)
    else:
        model = darknet.SCAFNet().to(device)
    model.load_state_dict(torch.load(opt.file))
    model.eval()
    model.requires_grad_(False)
    t2 = time.time()
    print('load model consume {:.2f} seconds'.format(t2-t1))
 

def predictPlain(model, img):
    '''
    model inference without Central Patch Cover Method
    ''' 
    # 0: parse configs
    conf_thres = opt.conf_thres
    nms_thres = opt.nms_thres 
    # 1: SCAF detections 
    d = {}    
    _, height, width = img.shape
    imgs = img.view([1, 3, height, width])
    # boxes with shape
    # [batch, boxes, 5]
    t1 = time.time()
    output = model(imgs)  
    t2 = time.time()
    print('process image consume {:.3f} seconds'.format(t2-t1))
    output = nmsTest(
        output, conf_thres=conf_thres, nms_thres=nms_thres)  
    return output[0]


def predictBox(model, img): 
    # counting time 
    t1 = time.time()
    # 0: parse configs
    conf_thres = opt.conf_thres
    nms_thres = opt.nms_thres
    overlap = opt.overlap
    # 1: pre setting 
    window_full_size = 512
    _, height, width = img.shape
    # window_full_size = min(height, width)
    window_full_size = opt.full_size
    window_center_size = window_full_size - 2*overlap
    assert height >= window_full_size and width >= window_full_size, 'resize image to {}x{}'.format(
        window_full_size, window_full_size)
    k, r = divmod((height-2*overlap), window_center_size)
    xmins = [i*window_center_size for i in range(k)]
    if r:
        xmins.append(height-window_full_size)
    k, r = divmod((width-2*overlap), window_center_size)
    ymins = [i*window_center_size for i in range(k)]
    if r:
        ymins.append(width-window_full_size)
    # 2: SCAF detections 
    d = {}  
    for xmin in xmins:
        # a patch is a batch
        for i, ymin in enumerate(ymins):
            imgs = img[:, xmin:xmin + window_full_size,
                       ymin:ymin+window_full_size]
            imgs = imgs.view([1, 3, window_full_size, window_full_size])
            # boxes with shape
            # [batch, boxes, 5]
            output = model(imgs)
            d[(xmin, ymin)] = output  
    t2 = time.time()
    print('model inference consume {:.2f} seconds'.format(t2-t1))
    # 3: Non Max Suppresion
    outputs = torch.zeros([0, 5])
    for k in d:
        xmin, ymin = k 
        output = d[k]  
        output = nmsTest(
            output, conf_thres=conf_thres, nms_thres=nms_thres) 
        output = output[0] 
        if output is None: continue 
        output[:, 0] += ymin
        output[:, 1] += xmin
        output[:, 2] += ymin
        output[:, 3] += xmin
        if output is not None:
            y = xyxy2xywh(output)
            outputs = torch.cat([outputs, y], 0)
    outputs = outputs.view([1, -1, 5])
    outputs = nmsTest(
        outputs, conf_thres=conf_thres, nms_thres=nms_thres)  
    t3 = time.time()
    print('non max suppress consume {:.2f} seconds'.format(t3-t2))
    return outputs[0]
 

def save2txt(short, boxes):
    ### 1. pre settings + ###
    save_path = 'mAP/input'
    gt_path = os.path.join(save_path, 'ground-truth')
    pred_path = os.path.join(save_path, 'detection-results')
    # make dirs
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)
    gt_name = os.path.join(gt_path, short[-3:]+'.txt')
    pred_name = os.path.join(pred_path, short[-3:]+'.txt')
    with open('data/boxes.json', 'r') as fp:
        d = json.load(fp)
    ### pre settings - ###
    ### 2. write gt to txt + ###
    ss = ''
    for box in d[short]:
        cx, cy, w, h = box
        xmin, ymin = cx-w/2, cy-h/2
        xmax, ymax = cx+w/2, cy+h/2
        tmp = [xmin, ymin, xmax, ymax]
        tmp = [str(int(x)) for x in tmp]
        # ss += 'car'+short[-3:]+' '
        ss += 'car' +' '
        ss += ' '.join(tmp)
        ss += '\n'
    with open(gt_name, 'w') as fp:
        fp.writelines(ss)
    ### write gt to txt - ###
    ### 3. write pred to txt + ###
    nums, _ = boxes.shape
    ss = ''
    cc = 0
    for i in range(nums):
        xmin, ymin, xmax, ymax, _ = np.array(boxes[i, :], dtype=np.int32)
        s = boxes[i, -1].item()
        tmp = [xmin, ymin, xmax, ymax]
        tmp = [str(int(x)) for x in tmp]
        # ss += 'car'+short[-3:]+' '
        ss += 'car' +' '
        ss += '{:.6f} '.format(s)
        ss += ' '.join(tmp)
        ss += '\n'
        cc += 1
        # 1K proposal
        if cc >= 1000:
            break
    with open(pred_name, 'w') as fp:
        fp.writelines(ss)
    ### write pred to txt - ###
    print('Done {} proposal for {} '.format(cc,short+'.JPG'))


def evaluate():
    # test all 5 images
    # report recall, precision and scene accuracy
    path = 'data/dlr'
    shorts = [ 
        '2012-04-26-Muenchen-Tunnel_4K0G0040',
        '2012-04-26-Muenchen-Tunnel_4K0G0080',
        '2012-04-26-Muenchen-Tunnel_4K0G0030',
        '2012-04-26-Muenchen-Tunnel_4K0G0051',
        '2012-04-26-Muenchen-Tunnel_4K0G0010'
    ]
    with open('data/boxes.json', 'r') as fp:
        boxes = json.load(fp)
    # report effective for each image
    for short in shorts:
        ############################ load image + #########################
        t1 = time.time()
        img = cv2.imread(os.path.join(path, short+'.JPG'),
                         cv2.IMREAD_UNCHANGED)
        if img is None:  continue 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)
        # bgr -> rgb -> tensor
        img = img.to(device)
        _, height, width = img.shape 
        t2 = time.time()
        print('load image {} consume {:.2f} seconds'.format(short+'.JPG', t2-t1))
        ############################ load image - #########################
        ############################ image process + #########################
        output = predictBox(model, img)   
        ############################ image process - #########################
        ############################ save boxes to txt +  ############################
        save2txt(short, output)
        ############################ save boxes to txt -  ############################
    print('All Done')


def eval_iou(full_size=3744):   
    evaluate()  
    rv = [] 
    a = b = 0 
    for iou in np.arange(0.5,1,0.05):
        ap = ap_iou(iou)
        a += ap 
        b += 1 
        rv.append([iou,ap])
    print('Done')
    print(" AP.5:.95 = {:.4f}".format(a/b))
    return rv 

def visIt(src_name, des_name, boxes, thres=0.5):
    '''
    get visual output 
    '''
    if boxes is None:
        return   
    img = cv2.imread(src_name, cv2.IMREAD_UNCHANGED)
    nums, _ = boxes.shape
    count = 0
    if opt.old_h==opt.new_h and opt.old_w==opt.new_w:
        scaleFactor = [-1] * 5
    else: 
        scaleFactor = [ opt.old_w / opt.new_w, opt.old_h / opt.new_h ,opt.old_w / opt.new_w, opt.old_h / opt.new_h , -1]
    for i in range(nums):
        if scaleFactor[0]<0:
            ymin, xmin, ymax, xmax, _ = np.array(boxes[i, :], dtype=np.int32)
        else: 
            ymin, xmin, ymax, xmax, _ = np.array([x*y for x,y in zip(boxes[i, :],scaleFactor)], dtype=np.int32) 
        s = boxes[i, -1].item()
        if s < thres:
            continue
        # draw
        line_color = (0, 255, 0) 
        # draw box: blue
        line_color = (238, 238, 141) 
        leftUp = (ymin,xmin)
        rightDown = (ymax,xmax)
        cv2.rectangle(img, leftUp, rightDown, line_color, 3)
        # draw cross: orange
        line_color = (0,165, 255) 
        cx, cy = int((xmax+xmin)//2), int((ymax+ymin)//2) 
        cv2.line(img,(cy-5,cx),(cy+5,cx),line_color,3)
        cv2.line(img,(cy,cx-5),(cy,cx+5),line_color,3)
        # draw digital: orange Red
        line_color = (0, 69, 255) 
        font=cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
        img = cv2.putText(img, ' {:.3f}'.format(s), (ymin, xmin), font, 0.8, line_color, 2) 
    print('{} has {} boxes'.format(src_name,nums))
    cv2.imwrite(des_name, img)

def gsdResample(height,width,gsd):
    x, y = height*gsd/12.5, width*gsd/12.5 
    height, width = int(np.ceil(x/32) * 32), int(np.ceil(y/32) * 32) 
    return height, width



def wholeBoxes(src_name,des_name): 
    print('start detection image: {}'.format(src_name)) 
    img = cv2.imread(src_name, cv2.IMREAD_UNCHANGED) 
    height, width, _ = img.shape   
    if 12<opt.gsd<11:
        pass 
        opt.new_h, opt.new_w = opt.old_h, opt.old_w = height, width 
    else:
        opt.old_h, opt.old_w = height, width 
        height, width = gsdResample(height,width,opt.gsd) 
        opt.new_h, opt.new_w = height, width 
        img = cv2.resize(img,(width,height),interpolation=cv2.INTER_AREA)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(img)
    img = img.to(device)  
    if height<2000 and width<2000:
        output = predictPlain(model, img)
    else:
        output = predictBox(model, img)
    # assert 0 
    # output = predictBox(model, img)
    # visConf(src_name,output,des_name)
    visIt(src_name,des_name, output)
    print('{} detection task done.'.format(des_name))


if __name__ == "__main__": 
    if opt.evaluation:
        eval_iou()
    else:
        src_path = opt.img_folder
        des_path = opt.result_folder
        for name in os.listdir(src_path):
            src_name = os.path.join(src_path,name)
            des_name = os.path.join(des_path,name)
            v = cv2.imread(src_name)
            if v is None:
                continue 
            wholeBoxes(src_name,des_name)  
    print('Done')
