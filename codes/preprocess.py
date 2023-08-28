import os
import logging
import argparse
import numpy as np
from numpy import matlib
import glob
import math

import scipy
from numpy import inf
import csv
import cv2
import shutil
import torch
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, RLock, shared_memory
from multiprocessing import current_process
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

# from tqdm_multiprocess import TqdmMultiProcessPool
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms

from utils import binary_mask_16
from utils import equalize_hist_16

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def generate_laplace_heatmap(gt, H, W, scale):
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
        gt,
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
    laplace_img = Image.fromarray(laplace_img).convert("L")

    return laplace_img


def generate_multivariate_laplace_2d(gt_points, H, W):
    correlation_coefficient = 0
    standard_deviation = 0.5
    k = 2
    v = (2 - k) / 2

    laplace_image = np.zeros((H, W))
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

        laplace_image += (
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

    laplace_image[laplace_image == inf] = laplace_image[laplace_image != inf].max() * 1.1
    laplace_image = laplace_image**2
    laplace_image = laplace_image / laplace_image.max()
    laplace_image *= 255
    laplace_image = Image.fromarray(laplace_image).convert("L")
    # # laplace_img = laplace_img.convert("RGB")

    return laplace_image


def generate_heatmap_multiprocessing(gt_file_path, heatmap_save_path, H, W, scale):
    laplace_image = None
    # heatmap 생성
    with open(gt_file_path, "r") as f:
        gt_points = list(csv.reader(f, delimiter=","))
        # str인 좌표를 int형으로 변경
        gt_points = [[eval(i[0]), eval(i[1])] for i in gt_points]
        # laplace_img = generate_laplace_heatmap(gt_points, H=H, W=W, scale=scale)
        laplace_image = generate_multivariate_laplace_2d(gt_points, H=H, W=W)
        f.close()

    laplace_image.save(heatmap_save_path)


def preprocess():
    """
    1. data path를 입력으로 받아서 해당 폴더 내의 데이터를 전처리
        1. laplace heatmap 생성
        2. (laplace heatmap과 원본 이미지를 patch로 분할)     # 추후 데이터 load할 때 적용해도 됨
    전처리한 데이터를 dataset 폴더에 기존 폴더명으로 저장
    train, validation 폴더에 나누지 않음(K-fold validation 위해서 dataset 클래스에서 불러오는게 편할듯)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="./original_data",
        help="path where the dataset folder is located",
    )
    parser.add_argument(
        "--data-folders",
        nargs="+",
        default=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
        help="list of dataset dirname",
    )
    parser.add_argument(
        "--dest-path",
        type=str,
        default="./dataset",
        help="path where the result image will be saved",
    )
    parser.add_argument("--scale", type=float, default=2.5, help="laplace heatmap scale factor")
    parser.add_argument("--height", type=int, default=3072)
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument(
        "--num-processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="number of processes",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="if the file already exists, do not overwrite",
    )

    args = parser.parse_args()
    logging.info(args)

    gt_file_list = []
    heatmap_save_path_list = []

    # 원본 이미지 복사, histogram 이미지 생성 및 저장
    # heatmap 이미지 저장 경로등은 여기서 구해놓고 나중에 multiprocessing으로 한 번에 처리
    for data_folder in args.data_folders:
        data_folder_path = os.path.join(args.data_path, data_folder)
        original_image_dir_path = os.path.join(args.dest_path, "original", data_folder)
        equalize_hist_image_dir_path = os.path.join(args.dest_path, "equalize_hist", data_folder)
        laplacian_image_dir_path = os.path.join(args.dest_path, "laplacian", data_folder)
        # heatmap_dir_path = os.path.join(args.dest_path, f"heatmap_scale_{args.scale:.1f}", data_folder)
        heatmap_dir_path = os.path.join(args.dest_path, f"heatmap_laplace", data_folder)

        os.makedirs(original_image_dir_path, exist_ok=True)
        os.makedirs(equalize_hist_image_dir_path, exist_ok=True)
        os.makedirs(laplacian_image_dir_path, exist_ok=True)
        os.makedirs(heatmap_dir_path, exist_ok=True)

        # 이미지 경로 리스트 생성
        tif_target = os.path.join(data_folder_path, "*.tif")
        tif_file_list = glob.glob(tif_target)
        tif_file_list.sort()

        # gt 경로 리스트 생성
        gt_txt_target = os.path.join(data_folder_path, "*.txt")
        cur_data_gt_file_list = glob.glob(gt_txt_target)

        # heatmap 저장 경로 저장
        cur_heatmap_save_path_list = []
        for gt_file_path in cur_data_gt_file_list:
            cur_heatmap_save_path_list.append(
                os.path.join(
                    heatmap_dir_path,
                    os.path.split(os.path.splitext(gt_file_path)[0])[1] + ".png",
                )
            )

        # gt 경로 리스트에 저장
        if not args.overwrite:
            new_gt_path_list = []
            new_heatmap_path_list = []
            for gt_path, heatmap_path in zip(cur_data_gt_file_list, cur_heatmap_save_path_list):
                if not os.path.exists(heatmap_path):
                    new_gt_path_list.append(gt_path)
                    new_heatmap_path_list.append(heatmap_path)

            gt_file_list += new_gt_path_list
            heatmap_save_path_list += new_heatmap_path_list
        else:
            gt_file_list += cur_data_gt_file_list
            heatmap_save_path_list += cur_heatmap_save_path_list

        # 원본 이미지 복사, 히스토그램 평활화 이미지 저장
        logging.info("원본 이미지 복사 및 히스토그램 평활화 이미지 저장")
        for tif_file_path in tqdm(tif_file_list):
            # 저장 파일명
            tif_filename = os.path.split(tif_file_path)[1]
            original_image_save_path = os.path.join(original_image_dir_path, tif_filename)
            equalize_hist_image_save_path = os.path.join(
                equalize_hist_image_dir_path,
                os.path.splitext(tif_filename)[0] + ".png",
            )
            laplacian_image_save_path = os.path.join(
                laplacian_image_dir_path,
                os.path.splitext(tif_filename)[0] + ".png",
            )

            # 파일이 이미 존재하는지 검사
            if bool(args.overwrite) != bool(not os.path.exists(original_image_save_path)):
                # 이미지 복사
                shutil.copy(tif_file_path, os.path.join(original_image_save_path))

            if bool(args.overwrite) != bool(not os.path.exists(equalize_hist_image_save_path)):
                # 입력 이미지 불러오기
                original_image = cv2.imread(tif_file_path, flags=cv2.IMREAD_UNCHANGED)

                # binary mask 적용
                binary_image = binary_mask_16(original_image, threshold=3000)
                # 히스토그램 평활화
                hist_image = equalize_hist_16(binary_image)

                # 히스토그램 평활화 이미지 저장
                cv2.imwrite(equalize_hist_image_save_path, hist_image)

            if bool(args.overwrite) != bool(not os.path.exists(laplacian_image_save_path)):
                # 입력 이미지 불러오기
                original_image = cv2.imread(tif_file_path, flags=cv2.IMREAD_UNCHANGED)
                # laplacian 이미지 생성
                laplacian_image = cv2.Laplacian(original_image, -1, ksize=3)
                laplacian_image = (
                    cv2.normalize(laplacian_image, None, 0, 65535, cv2.NORM_MINMAX, dtype=-1) * 65535
                )
                # laplacian 이미지 저장
                cv2.imwrite(laplacian_image_save_path, laplacian_image)

    logging.info("heatmap 생성 및 저장")
    results = process_map(
        generate_heatmap_multiprocessing,
        gt_file_list,
        heatmap_save_path_list,
        [args.height] * len(gt_file_list),
        [args.width] * len(gt_file_list),
        [args.scale] * len(gt_file_list),
        max_workers=args.num_processes,
        desc="overall progress",
    )


if __name__ == "__main__":
    preprocess()
