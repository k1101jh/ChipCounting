import os
import glob
import numpy as np
import csv
import cv2
import logging
import torch
import torch.nn as nn


def generate_heatmap(data_path: str, save_path: str, heatmap_size: int):
    def equalizeHist16(img):
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
        img = cv2.imread(tif_filepath, flags=cv2.IMREAD_UNCHANGED)
        print(img.max(), img.min())
        tif_size_list.append(img.size)
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = img
        # threshold_value, binary_mask = cv2.threshold(gray_img, 0, 65535, cv2.THRESH_OTSU)
        threshold_value, binary_mask = cv2.threshold(
            gray_img, 3000, 65535, cv2.THRESH_BINARY_INV
        )

        # cv2.imshow("binary", binary_mask)
        # cv2.waitKey()

        # 히스토그램 평활화
        # binary_img = binary_img * (65535 - binary_mask)
        gray_img = gray_img * binary_mask
        gray_img = equalizeHist16(gray_img)

        # binary 이미지에 heatmap 표시
        gray_gt_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        for gt_point in gt:
            gray_gt_img = cv2.circle(
                gray_gt_img, gt_point, heatmap_size, (0, 65535, 0), -1
            )

        # binary_gt 이미지 저장
        filename = os.path.split(os.path.splitext(tif_filepath)[0])[1]
        save_filepath = os.path.join(save_path, filename + f"_gt.png")
        logging.info(f"save binary image: {save_filepath}")
        cv2.imwrite(save_filepath, gray_gt_img)


if __name__ == "__main__":
    data_path = r"./Chipdata/data/D5"
    save_path = r"./code_test_results/binary_equalizeHist_images/D5"

    os.makedirs(save_path, exist_ok=True)

    generate_heatmap(data_path, save_path, heatmap_size=3)


# def generate_laplace_heatmap(gt, H, W, scale):
#     """laplace heatmap을 생성해서 반환하는 함수

#     앞으로 고민할 점: 육안으로 검은색인 영역의 실제 값은 0인가?
#                      (scale이 작으면 0일 것으로 생각됨)

#     Args:
#         gt (_type_): _description_
#         H (_type_): _description_
#         W (_type_): _description_
#         scale (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     x_axis_mtx = np.arange(W)
#     x_axis_mtx = np.matlib.repmat(x_axis_mtx, H, 1)
#     y_axis_mtx = np.arange(H)
#     y_axis_mtx = np.matlib.repmat(y_axis_mtx, W, 1).transpose()

#     laplace_img = np.zeros((H, W))
#     process_idx = current_process()._identity[0] - 1
#     for gt_point in tqdm(
#         gt,
#         desc=f"process {process_idx} generating heatmap...",
#         position=process_idx + 1,
#         leave=True,
#     ):
#         center_width_mtx = np.ones((H, W)) * gt_point[0]
#         center_height_mtx = np.ones((H, W)) * gt_point[1]
#         x_axis_dist = x_axis_mtx - center_width_mtx
#         y_axis_dist = y_axis_mtx - center_height_mtx
#         laplace_img += (1 / (2 * scale)) * np.exp(
#             -1 * (np.abs(x_axis_dist) + np.abs(y_axis_dist)) / (scale**2)
#         )

#     # x_linespace_mtx = torch.arange(-gt[0][0], W - gt[0][0], dtype=torch.float)
#     # y_linespace_mtx = torch.arange(-gt[0][1], H - gt[0][1], dtype=torch.float)
#     # x_axis_mtx, y_axis_mtx = torch.meshgrid(x_linespace_mtx, y_linespace_mtx)
#     # laplace_img = (1 / (2 * scale)) * torch.exp(
#     #     -1 * (torch.abs(x_axis_mtx) + torch.abs(y_axis_mtx)) / (scale**2)
#     # )
#     # x_axis_mtx = torch.arange(W, dtype=torch.float, device=device)
#     # x_axis_mtx = x_axis_mtx.repeat(H, 1)
#     # y_axis_mtx = torch.arange(H, dtype=torch.float, device=device)
#     # y_axis_mtx = y_axis_mtx.repeat(W, 1).transpose(0, 1)

#     # laplace_img = torch.zeros((H, W), device=device)

#     # for gt_point in gt:
#     #     center_width_mtx = torch.ones((H, W), dtype=torch.float, device=device) * gt_point[0]
#     #     center_height_mtx = torch.ones((H, W), dtype=torch.float, device=device) * gt_point[1]
#     #     x_axis_dist = (x_axis_mtx - center_width_mtx)
#     #     y_axis_dist = (y_axis_mtx - center_height_mtx)
#     #     laplace_img += (1 / (2 * scale)) * torch.exp(-1 * (torch.abs(x_axis_dist) + torch.abs(y_axis_dist)) / (scale ** 2))
#     #     break
#     laplace_img = laplace_img / laplace_img.max()
#     laplace_img *= 255
#     laplace_img = Image.fromarray(laplace_img)
#     # laplace_img = to_pil_image(laplace_img)
#     laplace_img = laplace_img.convert("RGB")

#     return laplace_img
