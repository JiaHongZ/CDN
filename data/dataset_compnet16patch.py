import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util

class DatasetDRNet(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """
    def __init__(self, n_channels=1, patch_size=120, sigma=25, isTrain=True, paths_H=None):
        super(DatasetDRNet, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.sigma = sigma
        self.sigma_test = self.sigma
        self.isTrain = isTrain
        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        if paths_H == None:
            # self.paths_H = util.get_image_paths(r"D:\zjh_home\data\trainsets\train5544")
            self.paths_H = util.get_image_paths(r"E:\image_denoising\datasets\DIV2Kpatch2")
        else:
            self.paths_H = util.get_image_paths(paths_H)

    def __getitem__(self, index):
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path

        if self.isTrain:
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = np.random.randint(0, 8)
            patch_H = util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            # --------------------------------
            # add noise
            # --------------------------------
            noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)
            img_L.add_(noise)
            # --------------------------------
            # clip
            # --------------------------------
            img_Ls = []
            img_Hs = []
            patchs = 4  # 4*4
            for m in range(patchs):
                for n in range(patchs):
                    img_Ls.append(img_L[:, m * int(self.patch_size // patchs):(m + 1) * int(self.patch_size // patchs),
                                  n * int(self.patch_size // patchs):(n + 1) * int(self.patch_size // patchs)])
                    img_Hs.append(img_H[:, m * int(self.patch_size // patchs):(m + 1) * int(self.patch_size // patchs),
                                  n * int(self.patch_size // patchs):(n + 1) * int(self.patch_size // patchs)])

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)
            h, w, c = img_L.shape
            # print(img_L.shape)
            if h % 8 != 0:
                img_L = img_L[:(h - h % 8), :, :]
                img_H = img_H[:(h - h % 8), :, :]
            if w % 8 != 0:
                img_L = img_L[:, :(w - w % 8),:]
                img_H = img_H[:, :(w - w % 8),:]
            # print(img_L.shape)
            # --------------------------------
            # add noise
            # --------------------------------
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_Ls = util.single2tensor3(img_L)
            img_Hs = util.single2tensor3(img_H)

        return img_Ls, img_Hs

    def __len__(self):
        return len(self.paths_H)
