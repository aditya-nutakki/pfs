import torch, torchvision
import os
import numpy
import torch.nn as nn
import torch.nn.functional as F
from helpers import *

image_dims = (3, 32, 32)

class DDPM(nn.Module):
    def __init__(self, beta1 = 10e-4, beta2 = 0.02, t = 1000, image_dims = image_dims) -> None:
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = t
    
        self.betas = torch.linspace(self.beta1, self.beta2, self.t)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim = -1)

        self.image_dims = image_dims
        self.c, self.h, self.w = image_dims




class UNet(nn.Module):
    def __init__(self, image_dims = image_dims) -> None:
        super().__init__()
        self.image_dims = image_dims
        self.c, self.h, self.w = image_dims
        self.hidden_down = 32
        self.max_pool = nn.MaxPool2d(2)

        self.down_block1 = DownBlock(self.c, self.hidden_down)
        self.down_block2 = DownBlock(self.hidden_down, self.hidden_down * 4)
        self.down_block3 = DownBlock(self.hidden_down * 4, self.hidden_down * 8)

        self.flatten = nn.Flatten()

        self.up = nn.Sequential(
            
        )


    def forward(self, x):
        x1 = self.down_block1(x)
        x1 = self.max_pool(x1)

        x2 = self.down_block1(x1)
        x2 = self.max_pool(x2)

        x3 = self.down_block1(x2)
        x3 = self.max_pool(x3)


        return x
        

if __name__ == "__main__":
    x = torch.randn(4, *image_dims)
    model = UNet()
    y = model(x)
    print(y, y.shape)