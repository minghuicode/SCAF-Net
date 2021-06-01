from __future__ import division

from models import darknet
from dataset import VehicleDataset

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def setup_seed(seed='GRSL2020'):
    if isinstance(seed, str):
        seed = hash(seed) % (2**32)
    else:
        seed = int(seed) % (2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    '''
    parse configs 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="size of each image batch")
    parser.add_argument("--adl_drop_rate", type=float, default=0.5,
                        help="drop rate of ADL module")
    parser.add_argument("--adl_drop_threshold", type=float, default=0.5,
                        help="drop threshold of ADL module") 
    parser.add_argument("--no_scene", action="store_true",
                        help="network without scene branch")
    parser.add_argument("--folder", type=str, default='data/dlr',
                        help="folder of training images")
    parser.add_argument("--gradient_accumulations", type=int,
                        default=4, help="number of gradient accums before step")
    parser.add_argument("--pretrained_weights", type=str,
                        help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=1,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=512,
                        help="size of each image dimension")
    parser.add_argument("--multiscale_training", default=True,
                        help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)


def train(model_name='scaf'):
    '''
    train a model from scratch 
    '''

    logtxt = 'checkpoints/mylog.txt' 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    with open(logtxt, 'w') as fp:
        fp.writelines('model log\n')

    # Setup random seed for model initial
    setup_seed('GRSL2020')

    # Initiate model
    if opt.no_scene:
        model = darknet.DetectNet(adl_drop_rate=opt.adl_drop_rate,
                                adl_drop_threshold=opt.adl_drop_threshold).to(device)
    else:
        model = darknet.SCAFNet(adl_drop_rate=opt.adl_drop_rate,
                                adl_drop_threshold=opt.adl_drop_threshold).to(device) 
    model.apply(weights_init_normal)
  
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        print('loading weights.......')
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = VehicleDataset(opt.folder, augment=True, img_size=opt.img_size,
                             multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    yolo_layers = [model.detect.small_vehicle.yolo,
                   model.detect.large_vehicle.yolo]
    iters = 0
    total_iters = opt.epochs * len(dataloader)
    start_time = time.time()
    pre = []
    pre100 = float('inf')
    for epoch in range(opt.epochs):
        model.train()
        for batch_i, (imgs, targets) in enumerate(dataloader):
            iters += 1

            try:
                imgs = Variable(imgs.to(device))
                if targets is None:
                    continue
                targets = Variable(targets.to(device), requires_grad=False)

                loss, _ = model(imgs, targets)
                loss_item = loss.item()
            except RuntimeError:
                loss_item = 20*pre100
            if loss_item > 10*pre100:
                # weak supervision need avoid wrong label or other situations
                pass
            else:
                pre.append(loss_item)
                pre = pre[-100:]
                if len(pre) > 80:
                    pre100 = sum(pre)/len(pre)

                loss.backward()

            if iters % opt.gradient_accumulations == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (
                epoch, opt.epochs, batch_i, len(dataloader))
 
            metric_table = [
                ["Layer", "Grid", "Branch Loss"]
            ]

            # Log metrics at each Head layer
            gg = model.detect.small_vehicle.yolo.metrics.get("grid_size",0)
            loss1 = model.detect.small_vehicle.yolo.metrics.get("loss",0)
            loss2 =  model.detect.large_vehicle.yolo.metrics.get("loss",0)
            loss3 = loss_item - loss1 - loss2 
            # small layer 
            small_metrics = [ "vehicle small", "%2d" % gg, "%.2f" % loss1]
            large_metrics = [ "vehicle large", "%2d" % (gg//2), "%.2f" % loss2]
            scene_metrics = [ "scene context", "%2d" % (gg//4), "%.2f" % loss3] 
            metric_table.append(small_metrics)
            metric_table.append(large_metrics)
            metric_table.append(scene_metrics) 

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss_item}"

            # Determine approximate time left for epoch
            epoch_total_left = total_iters - iters
            time_left = datetime.timedelta(
                seconds=epoch_total_left * (time.time() - start_time) / iters)
            log_str += f"\n---- ETA {time_left}"

            if loss_item > 2*pre100:
                print(log_str)
            else:
                print(log_str)
                with open(logtxt, 'a+') as fp:
                    fp.write(log_str)

    torch.save(model.state_dict(),  f"checkpoints/%s.pth" % model_name)


def show_fig(logtxt='checkpoints/mylog.txt'):
    import matplotlib.pyplot as plt
    loss = []
    with open(logtxt, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        if line[:10] != 'Total loss':
            continue
        v = float(line[10:])
        loss.append(v)
    plt.plot(loss[100:])
    plt.show()


if __name__ == "__main__":
    train()
