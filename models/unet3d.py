import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """
    Two consecutive 3D convolution blocks.
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


class UNet3D(nn.Module):

    def __init__(
        self,
        in_channels=4,
        num_classes=4
    ):

        super().__init__()

        # Encoder
        self.encoder1 = DoubleConv3D(
            in_channels,
            32
        )

        self.encoder2 = DoubleConv3D(
            32,
            64
        )

        self.encoder3 = DoubleConv3D(
            64,
            128
        )

        # Bottleneck
        self.bottleneck = DoubleConv3D(
            128,
            256
        )

        # Downsampling
        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        # Upsampling
        self.up3 = nn.ConvTranspose3d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.up2 = nn.ConvTranspose3d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.up1 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        # Decoder
        self.decoder3 = DoubleConv3D(
            256,
            128
        )

        self.decoder2 = DoubleConv3D(
            128,
            64
        )

        self.decoder1 = DoubleConv3D(
            64,
            32
        )

        # Final segmentation layer
        self.output = nn.Conv3d(
            32,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool(e1)
        )

        e3 = self.encoder3(
            self.pool(e2)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e3)
        )

        # Decoder
        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.decoder3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.decoder2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.decoder1(d1)

        return self.output(d1)