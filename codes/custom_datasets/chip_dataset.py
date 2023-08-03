import os
import random
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from utils import SplitTensorModule


class ChipDataset(Dataset):
    def __init__(self, data_dir_list, transform, **kwargs):
        """_summary_

        Args:
            data_dir_list (_type_): 사용할 데이터 폴더명 list. 예) ['D1', 'D3', 'D4']
            input_transform (_type_): transform 적용 이후 input image에만 적용하는 transform
            transform (_type_): input image와 heatmap에 모두 적용하는 transform
        """
        super.__init__(**kwargs)
        self.transform = transform
        self.tensor_split_module = SplitTensorModule(
            patch_size=kwargs["data"]["patch_size"],
            stride=kwargs["data"]["patch_stride"],
        )
        self.input_dataset = None
        self.heatmap_dataset = None

        image_folder_path_list = [
            os.path.join(kwargs["data"]["hist_image_dataset_path"], data_folder)
            for data_folder in data_dir_list
        ]
        heatmap_folder_path_list = [
            os.path.join(kwargs["data"]["heatmap_dataset_path"], data_folder)
            for data_folder in data_dir_list
        ]

        self.input_image_path_list = []
        self.heatmap_image_path_list = []
        for image_folder_path in image_folder_path_list:
            self.input_image_path_list += os.listdir(image_folder_path).sort()
        for heatmap_folder_path in heatmap_folder_path_list:
            self.heatmap_image_path_list += os.listdir(heatmap_folder_path).sort()

        assert len(self.input_image_path_list) == len(
            self.heatmap_image_path_list
        ), "입력 이미지와 heatmap 이미지 개수가 다릅니다."

        self.data_len = len(self.input_image_path_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        """
        현재는 patch를 나누고 transform 적용
        예상되는 문제점: patch에 Rotation, Crop, Shift등을 적용했을 때 발생하는 외곽의 검은 영역이 결과에 영향을 얼마나 주는가

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_image = Image.open(self.input_image_path_list[idx])
        heatmap_image = Image.open(self.heatmap_image_path_list[idx])

        # patch 나누기
        images = torch.cat((input_image, heatmap_image), 0)
        images, y_map, x_map = self.tensor_split_module(images)

        # random patch 선택
        random_idx = random.randrange(0, images.shape[0])
        patches = images[0, random_idx]

        # transform 적용
        patches = self.transform(patches[random_idx])

        return patches
