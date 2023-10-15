import torch, torchvision
from torchvision.transforms import transforms
import os, numpy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam

from helpers import *
import torchshow as ts
from torchvision.datasets.mnist import MNIST
from unet import UNet


import warnings
warnings.filterwarnings("ignore")

image_dims = (1, 32, 32)
c, h, w = image_dims
batch_size = 16
device = "cuda"


class DDPM(nn.Module):
    def __init__(self, beta1 = 10e-4, beta2 = 0.02, t = 1000, image_dims = image_dims, device = device) -> None:
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = t
    
        self.betas = torch.linspace(self.beta1, self.beta2, self.t)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim = -1)

        self.image_dims = image_dims
        self.c, self.h, self.w = image_dims

        self.model = UNet(input_channels = self.c, output_channels = self.c)
        self.criterion = nn.MSELoss()
        self.opt = Adam(self.model.parameters(), lr = 2e-4)


    def get_alphas(self, rand_steps):
        # returns alpha_cumprod at rand_step index
        return torch.Tensor(
            [self.alphas_cumprod[i.item()] for i in rand_steps]
        )


    def apply_noise(self, images, rand_steps):
        # images to be a tensor of (batch_size, c, h, w)
        noise = torch.randn(images.shape, device = device)
        noised_images = []

        for i in range(len(rand_steps)):
            alpha_hat = self.alphas_cumprod[rand_steps[i]]
            noised_images.append(
                images[i] * (alpha_hat ** 0.5) + ((1 - alpha_hat)**0.5) * noise[i]
            )

        noised_images = torch.stack(noised_images, dim = 0)
        return noised_images, noise


    def forward(self, images):        
        rand_steps = torch.randint(1, self.t, (batch_size, ))
        (xt, noise_t), (xt_minus_one, noise_t_minus_one) = self.apply_noise(images, rand_steps), self.apply_noise(images, rand_steps-1) # need to predict xt_minus_one given xt

        # print(f"rand step => {rand_step}")
        # ts.save(xt, "xts.jpeg")
        # ts.save(xt_minus_one, "xt-1s.jpeg")
        return xt, noise_t, xt_minus_one, noise_t_minus_one, rand_steps


    def reverse(self):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(batch_size, *image_dims, device = device)
            for _ in range(self.t):
                x = self.model(x)

        return x


    
def get_dataloader():
    dataset = MNIST(root="./", download=True,
                        transform=transforms.Compose([
                        transforms.Resize(w), # or h
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
    t = 512
    ddpm = DDPM(t = t)
    ddpm.model.to(device)
    epochs = 10

    for ep in range(epochs):
        for i, (images, _) in enumerate(dataloader):
            
            ddpm.opt.zero_grad()

            images = images.to(device)
            # xt, xt_minus_one, rand_steps = ddpm(images) # xt-1 is target given xt
            xt, noise_t, xt_minus_one, noise_t_minus_one, rand_steps = ddpm(images)
            
            xt, noise_t, rand_steps = xt.to(device), noise_t.to(device), rand_steps.to(device)
            pred_target = ddpm.model(xt, rand_steps)
            # pred_target is the "noise" predicted by the model; need this to be compared with "noise_t"

            loss = ddpm.criterion(pred_target, noise_t)
            # print(loss.item())
            
            loss.backward()
            ddpm.opt.step()

            if i % 100 == 0:
                print(loss.item())
            # for j in range(t):
            #     print(ddpm.alphas_cumprod[j])
            #     _images = apply_noise(images, ddpm.alphas_cumprod[j])
            #     if j % 4 == 0:
        # ts.save(xt, f"./{ep}_xt.jpeg")
        # ts.save(pred_target, f"./{ep}_preds.jpeg")
