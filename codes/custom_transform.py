import os
import glob
import numpy as np
import csv
import cv2
import logging
import torch
import torch.nn as nn


class BinaryMask16(object):
    def __init__(self, threshold=None):
        self.threshold = threshold

    def __call__(self, image):
        if self.threshold == None:
            threshold, binary_mask = cv2.threshold(image, 0, 65535, cv2.THRESH_OTSU)
        else:
            threshold, binary_mask = cv2.threshold(
                image, threshold, 65535, cv2.THRESH_BINARY_INV
            )

        return image * binary_mask


class EqualizeHist16(object):
    def __init__(self):
        pass

    def __call__(self, image):
        hist, bins = np.histogram(image.flatten(), 65536, [0, 65526])
        cdf = hist.cumsum()

        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 65535 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype("uint16")

        equ = cdf[image]
        return equ
