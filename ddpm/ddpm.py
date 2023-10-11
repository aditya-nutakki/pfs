import torch, torchvision
from torchvision.transforms import transforms
import os
import numpy
import torch.nn as nn
import torch.nn.functional as F
from helpers import *
import torchshow as ts
from torchvision.datasets.mnist import MNIST
from unet import UNet

import warnings
warnings.filterwarnings("ignore")

image_dims = (3, 32, 32)
c, h, w = image_dims
batch_size = 16


def apply_noise(images, alpha_t):
        # images to be a tensor of (batch_size, c, h, w)
        
        noise = torch.randn(images.shape)
        return images*(alpha_t**0.5) + ((1 - alpha_t)**0.5)*noise

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
    
    
def get_dataloader():
    dataset = MNIST(root="./", download=True,
                        transform=transforms.Compose([
                        transforms.Resize(w),
                        transforms.ToTensor(),
                        NormalizeToRange(-1, 1)
                        ]))

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


if __name__ == "__main__":
    # x = torch.randn(4, *image_dims)
    # model = UNet()
    # y = model(x)
    # print(y, y.shape)
    dataloader = get_dataloader()
    t = 128
    ddpm = DDPM()
    for i, (images, labels) in enumerate(dataloader):

        for j in range(t):
            print(ddpm.alphas_cumprod[j])
            _images = apply_noise(images, ddpm.alphas_cumprod[j])
            if j % 4 == 0:
                ts.save(_images, f"./{i}_{j}.jpeg")
        break
    