import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def thresh_func(mask, thresh=0.5):
    mask[mask >= thresh] = 1
    mask[mask < thresh] = 0


class mytensorsplit(nn.Module):
    def __init__(self, p_size=28, stri=10, pad=0, index=False, H=2400, W=2880):
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
        self.kernel_size = p_size
        self.unfold = nn.Unfold(kernel_size=p_size, stride=stri, padding=pad)
        Ymap, Xmap = np.mgrid[0:H:1, 0:W:1]
        self.Ymap, self.Xmap = torch.tensor(Ymap, dtype=torch.float).unsqueeze(
            0
        ), torch.tensor(Xmap, dtype=torch.float).unsqueeze(0)
        # [1, H, W]

    def forward(self, x):
        if self.index == True:
            x = self.unfold(x)
            ind_y = self.unfold(self.Ymap.view(1, 1, self.H, self.W))
            ind_x = self.unfold(self.Xmap.view(1, 1, self.H, self.W))
            x = (
                x.permute(0, 2, 1)
                .view(1, -1, self.kernel_size, self.kernel_size)
                .squeeze(0)
            )
            ind_y = (
                ind_y.permute(0, 2, 1)
                .view(1, -1, self.kernel_size, self.kernel_size)
                .squeeze(0)
            )
            ind_x = (
                ind_x.permute(0, 2, 1)
                .view(1, -1, self.kernel_size, self.kernel_size)
                .squeeze(0)
            )

            return x, ind_y, ind_x
        else:
            x = self.unfold(x)
            x = x.permute(0, 2, 1).view(1, -1, self.kernel_size, self.kernel_size)
            return x
        return 0


if __name__ == "__main__":
    ## tensor split
    inputs = torch.load("./codes/test_code/testinput.pt")
    print(inputs.shape)
    fun = mytensorsplit(p_size=512, stri=256, pad=0, index=True, H=2400, W=2880)
    patch, ymap, xmap = fun(inputs)
