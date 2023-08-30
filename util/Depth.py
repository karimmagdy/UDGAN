import numpy as np
import os, sys
import argparse, time
import torch
from torch.autograd import Variable
from torchvision.transforms import RandomHorizontalFlip
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from collections import Counter
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from models.model_fpn import I2D

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

class SLlog(nn.Module):
    def __init__(self):
        super(SLlog, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
            
        # filter out invalid pixels
        N = (real>0).float().sum()
        mask = (real==0)        
        fake[mask] = 1.
        real[mask] = 1.
        
        loss = 100.* torch.sum( torch.abs(torch.log(real)-torch.log(fake)) ) / N
        return loss

    
class RMSE_log(nn.Module):
    def __init__(self, use_cuda):
        super(RMSE_log, self).__init__()
        self.eps = 1e-8
        self.use_cuda = use_cuda
    
    def forward(self, fake, real):
        mask = real<1.
        n,_,h,w = real.size()
        fake = F.upsample(fake, size=(h,w), mode='bilinear')
        fake += self.eps

        N = len(real[mask])
        loss = torch.sqrt( torch.sum( torch.abs(torch.log(real[mask])-torch.log(fake[mask])) ** 2 ) / N )
        return loss

class iRMSE(nn.Module):
    def __init__(self):
        super(iRMSE, self).__init__()
        self.eps = 1e-8
    
    def forward(self, fake, real):
        n,_,h,w = real.size()
        fake = F.upsample(fake, size=(h,w), mode='bilinear')
        mask = real<1.
        n = len(real[mask])
        loss = torch.sqrt( torch.sum( torch.abs(1./real[mask] - 1./(fake[mask]+self.eps) ) ** 2 ) / n )
        return loss

def get_acc(output, target):
    # takes in two tensors to compute accuracy
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct_mask = pred.eq(target.data.view_as(pred))
    correct = correct_mask.cpu().sum()
    print("Target: ", Counter(target.data.cpu().numpy()))
    print("Pred: ", Counter(pred.cpu().numpy().flatten().tolist()))
    return float(correct)*100 / target.size(0) 
    
def get_depth(x):
  i2d = I2D(fixed_feature_weights=False)
  i2d = i2d.cuda()


#   reg_criterion = RMSE_log(use_cuda=True)
#   eval_metric = iRMSE()

  # resume

  # load_name = 'i2d_1_18.pth'
  # print("loading checkpoint %s" % (load_name))
#   state = i2d.state_dict()
  # checkpoint = torch.load(load_name)
  # state.update(checkpoint['model'])
#   i2d.load_state_dict(state)

  # del checkpoint
  # torch.cuda.empty_cache()

  # setting to train mode
  #i2d.eval()
  return i2d(x)

