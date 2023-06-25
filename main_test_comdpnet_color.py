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


sigma = 15
netname = 'color' # 'color' for color-scale; '' for gray-scale; 'realnoise' for real-world denoising
datasetname = ['set5','Kodak24']
batch_size = 1
patch_sizes = [128]
in_channels = 3 # 3 for color-scale; 1 for gray-scale
hiddens = 64

if netname == 'nonoisenet':
    from models.network_compdnet_nonoisenet import Net
elif netname == 'nocontrastive':
    from models.network_compdnet_nocontrastive import Net
elif netname == 'noimagenet':
    from models.network_compdnet_noimagenet import Net
elif netname == 'nossim':
    from models.network_compdnet import Net
elif netname == 'res':
    from models.network_compdnet_res import Net

lr = 0.00005
test_iter = 200
load_pretrained = True
logger_name = 'test'

save_path =  os.path.join('model_zoo',str(sigma)+netname)
if not os.path.exists(save_path):
    os.makedirs(save_path)
utils_logger.logger_info(logger_name, os.path.join('model_zoo',str(sigma)+netname, logger_name + '.log'))
logger = logging.getLogger(logger_name)

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
        H = util.tensor2uint(H)
        current_psnr = util.calculate_psnr(out, H, border=0)
        current_ssim = util.calculate_ssim(out, H, border=0)

        psnrs.append(current_psnr)
        ssim.append(current_ssim)

        avg_psnr += current_psnr
        avg_ssim += current_ssim

    avg_psnr = avg_psnr / count
    avg_ssim = avg_ssim / count

    return avg_psnr,avg_ssim,psnrs

if __name__ == '__main__':
    # total_loss_logger = VisdomPlotLogger('line', opts={'title': 'Loss'})
    # image_net_logger =  VisdomPlotLogger('line', opts={'title': 'Image net Loss'})
    # noise_net_logger = VisdomPlotLogger('line', opts={'title': 'Noise net Loss'})
    # psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train PSNR'})
    for datasetn in datasetname:
        test_set = DatasetDRNet(in_channels,patch_sizes[0],sigma,False,'testsets\\'+datasetn)
        test_loader = DataLoader(test_set, batch_size=1,
                                 shuffle=False, num_workers=0,
                                 drop_last=False, pin_memory=True)

        compnet = Net(in_channels,in_channels,hiddens).cuda()
        compnet = torch.nn.DataParallel(compnet)
        if load_pretrained:
            compnet.load_state_dict(torch.load(os.path.join(save_path,'best.pth')))

        best = 0
        count = 0
        test_val_, test_ssim, psnrs = test(compnet)
        logger.info('test PSNR:[{}] SSIM:[{}] iter:[{}]'.format(test_val_,test_ssim, count))
        # for i_psnr in range(len(psnrs)):
        #     logger.info('{:->4d}| {:<4.2f}dB'.format(i_psnr, psnrs[i_psnr]))