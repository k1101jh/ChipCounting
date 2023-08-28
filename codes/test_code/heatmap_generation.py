import os
import glob
import numpy as np
import csv
import cv2
import logging
import torch
import torch.nn as nn
import math
import scipy
from numpy import inf
from PIL import Image
from tqdm import tqdm
from multiprocessing import current_process
from tqdm.contrib.concurrent import process_map


def generate_laplace_heatmap(gt_points, H, W, scale):
    """laplace heatmap을 생성해서 반환하는 함수

    앞으로 고민할 점: 육안으로 검은색인 영역의 실제 값은 0인가?
                     (scale이 작으면 0일 것으로 생각됨)

    Args:
        gt (_type_): _description_
        H (_type_): _description_
        W (_type_): _description_
        scale (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_axis_mtx = np.arange(W)
    x_axis_mtx = np.matlib.repmat(x_axis_mtx, H, 1)
    y_axis_mtx = np.arange(H)
    y_axis_mtx = np.matlib.repmat(y_axis_mtx, W, 1).transpose()

    laplace_img = np.zeros((H, W))
    process_idx = current_process()._identity[0] - 1
    for gt_point in tqdm(
        gt_points,
        desc=f"process {process_idx} generating heatmap...",
        position=process_idx + 1,
        leave=True,
    ):
        center_width_mtx = np.ones((H, W)) * gt_point[0]
        center_height_mtx = np.ones((H, W)) * gt_point[1]
        x_axis_dist = x_axis_mtx - center_width_mtx
        y_axis_dist = y_axis_mtx - center_height_mtx
        laplace_img += (1 / (2 * scale)) * np.exp(
            -1 * (np.abs(x_axis_dist) + np.abs(y_axis_dist)) / (scale**2)
        )

    laplace_img = laplace_img / laplace_img.max()
    laplace_img *= 255
    laplace_img = Image.fromarray(laplace_img)
    # laplace_img = to_pil_image(laplace_img)
    laplace_img = laplace_img.convert("RGB")

    return laplace_img


def generate_multivariate_laplace_2d(gt_points, H, W):
    correlation_coefficient = 0
    standard_deviation = 0.5
    k = 2
    v = (2 - k) / 2

    laplace_img = np.zeros((H, W))
    process_idx = current_process()._identity[0] - 1
    for gt_point in tqdm(
        gt_points,
        desc=f"process {process_idx} generating heatmap...",
        position=process_idx + 1,
        leave=True,
    ):
        x_linespace_mtx = np.arange(-gt_point[0], W - gt_point[0])
        y_linespace_mtx = np.arange(-gt_point[1], H - gt_point[1])

        x_axis_mtx, y_axis_mtx = np.meshgrid(x_linespace_mtx, y_linespace_mtx)

        laplace_img += (
            (
                2
                / (
                    ((2 * math.pi) ** (k / 2))
                    * (standard_deviation**k)
                    * math.sqrt(1 - correlation_coefficient**2)
                )
            )
            * ((x_axis_mtx**2 + y_axis_mtx**2) / 2) ** (v / 2)
            * scipy.special.kve(
                v, np.sqrt(2 * (x_axis_mtx**2 + y_axis_mtx**2) / (1 - correlation_coefficient**2))
            )
        )

    # laplace_img[laplace_img == inf] = 10
    # laplace_img = laplace_img / laplace_img.max()
    laplace_img = laplace_img**2
    # laplace_img *= 255
    # laplace_img = Image.fromarray(laplace_img)
    # # laplace_img = laplace_img.convert("RGB")

    return laplace_img


if __name__ == "__main__":
    save_dir = r"./test_code_results/multivariate_laplace_2d/"
    save_path = os.path.join(save_dir, "D5_20191212103725.png")

    sample_image_path = r"./original_data/D5/20191212103725.tif"
    sample_gt_path = r"./original_data/D5/20191212103725.txt"

    H = 3072
    W = 3072
    scale = 2.5

    os.makedirs(save_dir, exist_ok=True)

    with open(sample_gt_path, "r") as f:
        gt_points = list(csv.reader(f, delimiter=","))
        # str인 좌표를 int형으로 변경
        gt_points = [[eval(i[0]), eval(i[1])] for i in gt_points]
        num_processes = 12
        max_num_elements_in_process = math.ceil(len(gt_points) / num_processes)
        gt_points_per_process = [
            gt_points[i : i + max_num_elements_in_process]
            for i in range(0, len(gt_points), max_num_elements_in_process)
        ]

        if len(gt_points_per_process) < num_processes:
            num_processes = len(gt_points_per_process)

        results = process_map(
            generate_multivariate_laplace_2d,
            gt_points_per_process,
            [H] * num_processes,
            [W] * num_processes,
            # [scale] * num_processes,
            max_workers=num_processes,
            desc="overall progress",
        )

        result_image = np.zeros((H, W))
        for result in results:
            result_image += result

        result_image[result_image == inf] = result_image[result_image != inf].max() * 1.1
        result_image = result_image / result_image.max()
        result_image *= 255
        result_image = Image.fromarray(result_image).convert("L")

        result_image.save(save_path)
