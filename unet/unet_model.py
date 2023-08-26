import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop

image_dims = (1, 572, 572) # c, h, w
num_classes = 2

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


class Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        return self.pool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 2) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.up(x)
        # print(x.shape)
        return F.relu(x)


class CropConcat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, down_x, x):
        return torch.cat([x, center_crop(down_x, [x.shape[2], x.shape[3]])], dim = 1)


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down1 = DoubleConv(image_dims[0], 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.down5 = DoubleConv(512, 1024)

        self.pool = Pool()
        self.cc = CropConcat()

        self.up4 = UpSample(1024, 512)
        self.up4_d = DoubleConv(1024, 512)

        self.up3 = UpSample(512, 256)
        self.up3_d = DoubleConv(512, 256)

        self.up2 = UpSample(256, 128)
        self.up2_d = DoubleConv(256, 128)

        self.up1 = UpSample(128, 64)
        self.up1_d = DoubleConv(128, 64)

        self.conv = nn.Conv2d(64, num_classes, kernel_size = 1)

    def forward(self, x):
        # print(f"input shape => {x.shape}")
        x1 = F.relu(self.pool(self.down1(x)))
        # print(f"x1 shape => {x1.shape}")
        x2 = F.relu(self.pool(self.down2(x1)))
        # print(f"x2 shape => {x2.shape}")
        x3 = F.relu(self.pool(self.down3(x2)))
        # print(f"x3 shape => {x3.shape}")
        x4 = F.relu(self.pool(self.down4(x3)))
        # print(f"x4 shape => {x4.shape}")
        x5 = F.relu(self.down5(x4)) # make maxpool2d a separate thing to implement
        # print(f"x5 shape => {x5.shape}")

        # upsampling
        # print("upsampling")
        upx4 = self.up4(x5)
        upx4 = self.cc(x4, upx4)
        upx4 = self.up4_d(upx4)
        # print(f"up4 shape => {upx4.shape}")
        
        upx3 = self.up3(upx4)
        upx3 = self.cc(x3, upx3)
        upx3 = self.up3_d(upx3)
        # print(f"up3 shape => {upx3.shape}")

        upx2 = self.up2(upx3)
        upx2 = self.cc(x2, upx2)
        upx2 = self.up2_d(upx2)
        # print(f"up2 shape => {upx2.shape}")

        upx1 = self.up1(upx2)
        upx1 = self.cc(x1, upx1)
        upx1 = self.up1_d(upx1)
        # print(f"up1 shape => {upx1.shape}")

        return self.conv(upx1) # returning only the logits so that this can be passed into CrossEntropyLoss directly


if __name__ == "__main__":
    x = torch.randn(image_dims).unsqueeze(dim=0)
    unet = UNet()
    x = unet(x)
    print(x.shape)