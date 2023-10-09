import torch, torchvision
import os
import numpy
import torch.nn as nn
import torch.nn.functional as F


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

        self.down = nn.Sequential(
            nn.Conv2d(self.c, self.hidden_down, kernel_size=3),
            nn.BatchNorm2d(self.hidden_down),
            nn.ReLU(),

            nn.Conv2d(self.hidden_down, self.hidden_down * 2, kernel_size=3),
            nn.BatchNorm2d(self.hidden_down * 2),
            nn.ReLU(),

            nn.Conv2d(self.hidden_down * 2, self.hidden_down * 4, kernel_size=3),
            nn.BatchNorm2d(self.hidden_down),
            nn.ReLU()

        )

        self.flatten = nn.Flatten()

        self.up = nn.Sequential(
            
        )


    def forward(self, x):
        x = self.down(x)
        x = self.flatten(x)
        return self.up(x)
        

if __name__ == "__main__":
    x = torch.randn(4, *image_dims)
    model = UNet()
    y = model(x)
    print(y, y.shape)