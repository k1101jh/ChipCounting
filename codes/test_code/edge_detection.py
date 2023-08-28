import os
import glob
import numpy as np
import csv
import cv2
import logging


def edge_detection(data_path, save_path, heatmap_size=3):
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

    # 이미지 읽어서 edge detection 적용하기
    tif_size_list = []
    for tif_filepath, gt in zip(tif_file_list, gt_list):
        image = cv2.imread(tif_filepath, flags=cv2.IMREAD_UNCHANGED)
        print(image.max(), image.min())
        tif_size_list.append(image.size)

        sobel = cv2.Sobel(image, -1, 1, 1, 3)
        laplacian = cv2.Laplacian(image, -1, ksize=3)
        canny = cv2.Canny((image / 256).astype("uint8"), 20, 50)

        laplacian = cv2.normalize(laplacian, None, 0, 65535, cv2.NORM_MINMAX, dtype=-1) * 65535
        # laplacian = laplacian.astype("uint16") * 255
        sobel = equalize_hist_16(sobel)
        # laplacian = equalize_hist_16(laplacian)

        # cv2.imshow("sobel", sobel)
        cv2.imshow("laplacian", laplacian)
        # cv2.imshow("canny", canny)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # edge 이미지에 heatmap 표시
        sobel_gt_img = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
        laplacian_gt_img = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        canny_gt_img = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        for gt_point in gt:
            sobel_gt_img = cv2.circle(sobel_gt_img, gt_point, heatmap_size, (0, 65535, 0), -1)
            laplacian_gt_img = cv2.circle(laplacian_gt_img, gt_point, heatmap_size, (0, 65535, 0), -1)
            canny_gt_img = cv2.circle(canny_gt_img, gt_point, heatmap_size, (0, 255, 0), -1)

        # edge 이미지 저장
        filename = os.path.split(os.path.splitext(tif_filepath)[0])[1]
        save_filepath_sobel = os.path.join(save_path, filename + f"_sobel_gt.png")
        save_filepath_laplacian = os.path.join(save_path, filename + f"_laplacian_gt.png")
        save_filepath_canny = os.path.join(save_path, filename + f"_canny_gt.png")
        cv2.imwrite(save_filepath_sobel, sobel_gt_img)
        cv2.imwrite(save_filepath_laplacian, laplacian_gt_img)
        cv2.imwrite(save_filepath_canny, canny_gt_img)


if __name__ == "__main__":
    data_path = r"./original_data/D5"
    save_path = r"./test_code_results/edge_detection_images/D5"

    os.makedirs(save_path, exist_ok=True)

    edge_detection(data_path, save_path, heatmap_size=3)
