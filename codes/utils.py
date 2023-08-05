import os
import glob
import numpy as np
import csv
import cv2
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def binary_mask_16(image, threshold=None):
    if threshold == None:
        threshold_value, binary_mask = cv2.threshold(image, 0, 65535, cv2.THRESH_OTSU)
    else:
        threshold_value, binary_mask = cv2.threshold(
            image, threshold, 1, cv2.THRESH_BINARY_INV
        )
    return image * binary_mask


def equalize_hist_16(image):
    hist, bins = np.histogram(image.flatten(), 65536, [0, 65526])
    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 65535 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint16")

    result = cdf[image]
    return result


def plot_images(inp, title=None, show=True, save_path=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)

    if title:
        plt.title(title)

    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path)

    plt.close()


class SplitTensorModule(nn.Module):
    def __init__(self, patch_size=28, stride=10, pad=0, index=False, H=2400, W=2880):
        """
        x를 [-1, psize*psize, patch 개수]로 나눈 뒤
        [-1, patch 개수, psize, psize]로 변형

        오른쪽과 아래쪽에 남는 부분이 patch에 포함이 되지 않을 수 있다.

        Args:
            p_size (int, optional): _description_. Defaults to 28.
            stri (int, optional): _description_. Defaults to 10.
            pad (int, optional): _description_. Defaults to 0.
            index (bool, optional): _description_. Defaults to False.
            H (int, optional): _description_. Defaults to 2400.
            W (int, optional): _description_. Defaults to 2880.
        """
        super().__init__()
        self.H = H
        self.W = W
        self.index = index
        self.kernel_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride, padding=pad)
        Ymap, Xmap = np.mgrid[0:H:1, 0:W:1]
        self.Ymap = torch.tensor(Ymap, dtype=torch.float).unsqueeze(0)
        self.Xmap = torch.tensor(Xmap, dtype=torch.float).unsqueeze(0)
        # [1, H, W]

    def forward(self, x):
        if self.index == True:
            x = self.unfold(x)
            ind_y = self.unfold(self.Ymap.view(1, 1, self.H, self.W))
            ind_x = self.unfold(self.Xmap.view(1, 1, self.H, self.W))
            x = x.permute(0, 2, 1).view(
                x.shape[0], -1, self.kernel_size, self.kernel_size
            )
            ind_y = ind_y.permute(0, 2, 1).view(
                1, -1, self.kernel_size, self.kernel_size
            )
            ind_x = ind_x.permute(0, 2, 1).view(
                1, -1, self.kernel_size, self.kernel_size
            )

            return x, ind_y, ind_x
        else:
            x = self.unfold(x)
            x = x.permute(0, 2, 1).view(
                x.shape[0], -1, self.kernel_size, self.kernel_size
            )
            return x
