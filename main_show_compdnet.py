import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from data.dataset_compnet import DatasetDRNet
import matplotlib.pyplot as plt
from models.network_compdnet import Net

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
import torch
import torch.nn as nn
from models.loss_ssim import SSIMLoss
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchnet.logger import VisdomPlotLogger, VisdomLogger

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


sigma = '50'
netname = 'color'
datasetname = ['set5']
batch_size = 1
patch_sizes = [128]
in_channels = 3
hiddens = 64

if netname == 'nonoisenet':
    from models.network_compdnet_nonoisenet import Net
elif netname == 'nocontrastive':
    from models.network_compdnet_nocontrastive import Net
elif netname == 'noimagenet':
    from models.network_compdnet_noimagenet import Net
elif netname == 'realnoise':
    from models.network_compdnet import Net
else:
    from models.network_compdnet import Net


lr = 0.0001
test_iter = 200
load_pretrained = True
save_path =  os.path.join('model_zoo',str(sigma)+netname)

seed = 100
# logger.info('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def test(model):
    model.eval()
    avg_psnr = 0
    avg_ssim = 0
    psnrs = []
    ssim = []
    count = 0
    for i, data in enumerate(test_loader):
        count += 1
        L, H = data
        L = L.cuda()
        H = H.cuda()
        with torch.no_grad():
            out = compnet(x=L,train = False)
            out = util.tensor2uint(out)
        L = util.tensor2uint(L)
        util.imsave(L, str(i)+'_N.png')
        util.imsave(out, str(i)+'.png')

    return avg_psnr,avg_ssim,psnrs

if __name__ == '__main__':
    # total_loss_logger = VisdomPlotLogger('line', opts={'title': 'Loss'})
    # image_net_logger =  VisdomPlotLogger('line', opts={'title': 'Image net Loss'})
    # noise_net_logger = VisdomPlotLogger('line', opts={'title': 'Noise net Loss'})
    # psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train PSNR'})
    for datasetn in datasetname:
        test_set = DatasetDRNet(in_channels,patch_sizes[0],int(sigma),False,'testsets\\'+datasetn)
        test_loader = DataLoader(test_set, batch_size=1,
                                 shuffle=False, num_workers=0,
                                 drop_last=False, pin_memory=True)

        compnet = Net(in_channels,in_channels,hiddens).cuda()
        compnet = torch.nn.DataParallel(compnet)
        if load_pretrained:
            compnet.load_state_dict(torch.load(os.path.join(save_path,'best.pth')))

        best = 0
        count = 3200
        test_val_, test_ssim, psnrs = test(compnet)
