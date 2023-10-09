import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, normalize = True, act = "relu") -> None:
        super().__init__()
        self.normalize = normalize
        self.act = nn.ReLU() if act == "relu" else nn.SiLU()
        self.down_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, stride = stride, padding = padding),
            nn.Conv2d(out_channels, out_channels, kernel_size= kernel_size, stride = stride, padding = padding),
            self.act
        )
        self.batch_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.batch_norm(x) if self.normalize else x
        return self.down_block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 2, padding = 0) -> None:
        super().__init__()

        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, stride=2, padding = 0)
        )


    def forward(self, x, down_x):
        return self.up_block(x, down_x)


# image_dims = (1, 28, 28)
# if __name__ == "__main__":
#     x = torch.randn(4, *image_dims)
#     model = DownBlock(1, 12, act = "silu")
#     y = model(x)
#     print(y, y.shape)