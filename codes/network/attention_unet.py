import torch.nn as nn
import torch


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super.__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super.__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, num_coefficients):
        super.__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g, num_coefficients, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm2d(num_coefficients),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l, num_coefficients, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm2d(num_coefficients),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                num_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=None):
        super.__init__()
        assert len(channels) == 5, "Unet channel 개수가 구현과 다릅니다. config 파일을 확인하세요."

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DoubleConvBlock(in_channels=in_channels, out_channels=channels[0])
        self.conv2 = DoubleConvBlock(
            in_channels=channels[0],
            out_channels=channels[1],
        )
        self.conv3 = DoubleConvBlock(
            in_channels=channels[1],
            out_channels=channels[2],
        )
        self.conv4 = DoubleConvBlock(
            in_channels=channels[2],
            out_channels=channels[3],
        )
        self.conv5 = DoubleConvBlock(
            in_channels=channels[3],
            out_channels=channels[4],
        )

        self.up4 = UpsampleBlock(in_channels=channels[4], out_channels=channels[3])
        self.attention4 = AttentionBlock(
            F_g=channels[3], F_l=channels[3], num_coefficients=channels[3] // 2
        )
        self.up_conv4 = DoubleConvBlock(
            in_channels=channels[4], out_channels=channels[3]
        )

        self.up3 = UpsampleBlock(in_channels=channels[3], out_channels=channels[2])
        self.attention3 = AttentionBlock(
            F_g=channels[2], F_l=channels[2], num_coefficients=channels[2] // 2
        )
        self.up_conv3 = DoubleConvBlock(
            in_channels=channels[3], out_channels=channels[2]
        )

        self.up2 = UpsampleBlock(in_channels=channels[2], out_channels=channels[1])
        self.attention2 = AttentionBlock(
            F_g=channels[1], F_l=channels[1], num_coefficients=channels[1] // 2
        )
        self.up_conv2 = DoubleConvBlock(
            in_channels=channels[2], out_channels=channels[1]
        )

        self.up1 = UpsampleBlock(in_channels=channels[1], out_channels=channels[0])
        self.attention1 = AttentionBlock(
            F_g=channels[0], F_l=channels[0], num_coefficients=channels[0] // 2
        )
        self.up_conv1 = DoubleConvBlock(
            in_channels=channels[1], out_channels=channels[0]
        )

        self.last_conv = nn.Conv2d(
            channels[0], out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.maxpool(x)
        x2 = self.conv2(x)

        x3 = self.maxpool(x)
        x3 = self.conv3(x)

        x4 = self.maxpool(x)
        x4 = self.conv4(x)

        x5 = self.maxpool(x)
        x5 = self.conv5(x)

        d5 = self.up4(x5)
        s4 = self.attention4(d5, skip_connection=x4)
        d4 = torch.cat((s4, d5), dim=1)
        d4 = self.up_conv4(d4)

        d4 = self.up3(x)
        s3 = self.attention3(d4, skip_connection=x3)
        d3 = torch.cat((s3, d4), dim=1)
        d3 = self.up_conv3(d3)

        d3 = self.up2(x)
        s2 = self.attention2(d3, skip_connection=x2)
        d2 = torch.cat((s2, d3), dim=1)
        d2 = self.up_conv2(d2)

        d2 = self.up1(x)
        s1 = self.attention1(d2, skip_connection=x1)
        d1 = torch.cat((s1, d2), dim=1)
        d1 = self.up_conv1(d1)

        out = self.last_conv(d1)

        return out
