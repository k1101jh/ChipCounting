import os
import glob
import numpy as np
import csv
import cv2
import logging
import torch
import torch.nn as nn


def generate_input_image(data_path: str, save_path: str, heatmap_size: int):
    def equalize_hist_16(img):
        hist, bins = np.histogram(img.flatten(), 65536, [0, 65526])
        cdf = hist.cumsum()

        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 65535 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype("uint16")

        equ = cdf[img]
        return equ

    tif_target = os.path.join(data_path, "*.tif")
    gt_txt_target = os.path.join(data_path, "*.txt")

    tif_file_list = glob.glob(tif_target)
    gt_file_list = glob.glob(gt_txt_target)

    tif_file_list.sort()
    gt_file_list.sort()

    # txt 파일 읽어서 gt 데이터 확인하기
    gt_list = []
    for gt_filepath in gt_file_list:
        # gt_df_list.append(pd.read_csv(gt_filename, sep=','))
        with open(gt_filepath, "r") as f:
            gt_points = list(csv.reader(f, delimiter=","))
            # str인 좌표를 int형으로 변경
            gt_points = [[eval(i[0]), eval(i[1])] for i in gt_points]
            gt_list.append(gt_points)
            f.close()

    # 이미지 읽어서 이진화 적용하기
    tif_size_list = []
    for tif_filepath, gt in zip(tif_file_list, gt_list):
        image = cv2.imread(tif_filepath, flags=cv2.IMREAD_UNCHANGED)
        print(image.max(), image.min())
        tif_size_list.append(image.size)
        gray_img = image
        # threshold_value, binary_mask = cv2.threshold(gray_img, 0, 65535, cv2.THRESH_OTSU)
        threshold_value, binary_mask = cv2.threshold(gray_img, 3000, 1, cv2.THRESH_BINARY_INV)

        # 히스토그램 평활화
        # binary_img = binary_img * (1 - binary_mask)
        gray_img = gray_img * binary_mask
        gray_img = equalize_hist_16(gray_img)

        # binary 이미지에 좌표 표시
        gray_gt_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        for gt_point in gt:
            gray_gt_img = cv2.circle(gray_gt_img, gt_point, heatmap_size, (0, 65535, 0), -1)

        # binary_gt 이미지 저장
        filename = os.path.split(os.path.splitext(tif_filepath)[0])[1]
        save_filepath = os.path.join(save_path, filename + f"_gt.png")
        logging.info(f"save binary image: {save_filepath}")
        cv2.imwrite(save_filepath, gray_gt_img)


if __name__ == "__main__":
    data_path = r"./original_data/D5"
    save_path = r"./test_code_results/binary_equalize_hist_images/D5"

    os.makedirs(save_path, exist_ok=True)

    generate_input_image(data_path, save_path, heatmap_size=3)
