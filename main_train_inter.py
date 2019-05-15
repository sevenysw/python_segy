# -*- coding: utf-8 -*-
import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from get_patch import *


# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--data_dir', default='data/train', type=str, help='path of train data')
parser.add_argument('--rate', default=2, type=int, help='sampling rate')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--patch_size', default=(40,40), type=int, help='patch size')
parser.add_argument('--stride', default=(32,32), type=int, help='the step size to slide on the data')
parser.add_argument('--jump', default=3, type=int, help='the space between shot')
parser.add_argument('--download', default=False, type=bool, help='if you will download the dataset from the internet')
parser.add_argument('--datasets', default = 0, type = int, help='the num of datasets you want be download,if download = True')
parser.add_argument('--train_data_num', default=100000, type=int, help='the num of the train_data')
parser.add_argument('--aug_times', default=0, type=int, help='Number of aug operations')
parser.add_argument('--scales', default=[1], type=list, help='data scaling')
parser.add_argument('--agc', default=True, type=int, help='Normalize each trace by amplitude')
parser.add_argument('--verbose', default=True, type=int, help='Whether to output the progress of data generation')
parser.add_argument('--display', default=1000, type=int, help='interval for displaying loss')

args = parser.parse_args()
batch_size = args.batch_size
cuda = torch.cuda.is_available()
torch.set_default_dtype(torch.float64)

n_epoch = args.epoch
rate = args.rate


if not os.path.exists('models_inter'):
    os.mkdir('models_inter')

save_dir = os.path.join('models_inter', args.model+'_' + 'rate' + str(rate))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


class DnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.dncnn(x)
        return out

    def _initialize_weights(self):

        print('init weight')    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)



def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DnCNN()
    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 1:
        print('resuming by loading epoch %03d\n' % (initial_epoch-1))
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        if initial_epoch >= n_epoch:
            print("training have finished")

        else:
            model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % (initial_epoch-1)))

    model.train()
    criterion = nn.MSELoss(reduce = True,size_average = False)
    if cuda:
        model = model.cuda()
         # device_ids = [0]
         # model = nn.DataParallel(model, device_ids=device_ids).cuda()
         # criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)
    # learning rates

    xs = datagenerator(data_dir = args.data_dir,patch_size = args.patch_size,stride = args.stride,train_data_num = args.train_data_num,
        download=args.download,datasets =args.datasets,aug_times=args.aug_times,scales = args.scales,verbose=args.verbose,jump=args.jump,agc=args.agc)
    xs = xs.astype(np.float64)
    
    xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))

    for epoch in range(initial_epoch, n_epoch):

        scheduler.step(epoch)  # step to the learning rate in this epcoh

        # tensor of the clean patches, NXCXHXW

        # DDataset = DenoisingDataset(xs, sigma) 
        DDataset = DownsamplingDataset(xs, rate,regular = True) 
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()
        for n_count, batch_yxm in enumerate(DLoader):
            optimizer.zero_grad()
            if cuda:
                #batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
                batch_x, batch_y,mask = batch_yxm[1].cuda(), batch_yxm[0].cuda(),batch_yxm[2].cuda()
            loss = criterion(model(batch_y),batch_x)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if n_count % args.display == 0:
                print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))






