import torch, torchvision
from torchvision.transforms import transforms
import os, numpy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam

from helpers import *
import torchshow as ts
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
# from unet import UNet
# from unet_time import UNet
# from unet_attn import UNet
from unet_lr import Unet
from time import time

import warnings
warnings.filterwarnings("ignore")

image_dims = (3, 128, 128)
# image_dims = (1, 32, 32)
c, h, w = image_dims
batch_size = 8
lr = 1e-4
device = "cuda"


save_dir = "./models"
os.makedirs(save_dir, exist_ok = True)



class DDPM(nn.Module):
    def __init__(self, beta1 = 10e-4, beta2 = 0.02, t = 1000, image_dims = image_dims, device = device, down_factor = 1) -> None:
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = t
    
        # self.betas = torch.linspace(self.beta1, self.beta2, self.t) # linear schedule
        self.betas = self.get_cosine_betas(timesteps = self.t) # cosine schedule

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim = -1)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.betas_hat = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


        self.image_dims = image_dims
        self.c, self.h, self.w = image_dims

        # self.model = UNet(input_channels = self.c, output_channels = self.c, down_factor = down_factor)
        # self.model = UNet(input_channels = self.c, output_channels = self.c)
        self.model = Unet(dim = 64, dim_mults=(1, 2, 4, 8), flash_attn = False)

        self.criterion = nn.MSELoss()
        self.opt = Adam(self.model.parameters(), lr = lr)

        self.save_interval = 256


    def get_cosine_betas(self, timesteps, s = 0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)




    def get_alphas(self, rand_steps):
        # returns alpha_cumprod at rand_step index
        return torch.Tensor(
            [self.alphas_cumprod[i.item()] for i in rand_steps]
        )


    def apply_noise(self, images, rand_steps):
        # images to be a tensor of (batch_size, c, h, w)
        bs = images.shape[0]
        noise = torch.randn(images.shape, device = device)
        noised_images = []

        # print(images.shape, noise.shape)
        # for i in range(len(rand_steps)):
        for i in range(bs):
            alpha_hat = self.alphas_cumprod[rand_steps[i]]
            noised_images.append(
                images[i] * (alpha_hat ** 0.5) + ((1 - alpha_hat)**0.5) * noise[i]
            )

        noised_images = torch.stack(noised_images, dim = 0)
        return noised_images, noise


    def forward(self, images):        
        rand_steps = torch.randint(0, self.t, (batch_size, ))
        (xt, noise_t), (xt_minus_one, noise_t_minus_one) = self.apply_noise(images, rand_steps), self.apply_noise(images, rand_steps-1) # need to predict xt_minus_one given xt

        # visualising
        # print(f"rand step => {rand_steps[3]}")
        # ts.save(xt, "xts.jpeg")
        # ts.save(xt_minus_one, "xt-1s.jpeg")
        
        return xt, noise_t, xt_minus_one, noise_t_minus_one, rand_steps


    def reverse(self, ep, rev_batch_size = batch_size):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(rev_batch_size, *image_dims, device = device)
            # denoise image here
            print("Denoising ...")
            stime = time()
            for t in range(self.t - 1, -1, -1):
                
                noise_comp = torch.randn(rev_batch_size, *image_dims, device=device)*torch.sqrt(self.betas_hat[t]) # can multiply it with torch.sqrt(self.betas[t]) as well and you would get similar results
                one_by_alpha = (1/torch.sqrt(self.alphas[t]))
                _t = torch.Tensor([t]).type(torch.LongTensor).to(device)
                x = (one_by_alpha * (x - ((self.betas[t]) * self.model(x, _t) )/(torch.sqrt(1 - self.alphas_cumprod[t])))) + noise_comp

                if t % self.save_interval == 0:
                    ts.save(x, f"diffusion_{t}_{ep}.jpeg")
            ftime = time()

            print(f"reverse diffusion done in {ftime-stime}s for {self.t} time steps")

    
def get_dataloader(dataset_type = "mnist"):
    print(f"LOADING {dataset_type} ...")
    if dataset_type == "mnist":
        dataset = MNIST(root="./", download=True,
                        transform=transforms.Compose([
                        transforms.Resize(w), # or h
                        transforms.ToTensor(),
                        NormalizeToRange(-1, 1)
                        ])) 
        
    elif dataset_type == "cifar":
        dataset = CIFAR10(root="./", download=True,
                        transform=transforms.Compose([
                        transforms.Resize(w), # or h
                        transforms.ToTensor(),
                        NormalizeToRange(-1, 1)
                        ])) 
        
    else:
        dataset = BikesDataset("/mnt/d/work/datasets/bikes/white_bg", img_sz = 128, limit = -1)

    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = True)


if __name__ == "__main__":
    
    dataloader = get_dataloader(dataset_type="custom")
    t = 512
    ddpm = DDPM(t = t, down_factor = 1)
    ddpm.model.to(device)
    epochs = 1000

    print(f"Training on {sum(p.numel() for p in ddpm.model.parameters())} parameters ... ")

    for ep in range(epochs):
        ddpm.model.train()
        
        losses = []
        # for i, (images, _) in enumerate(dataloader):
        for i, images in enumerate(dataloader):
            ddpm.opt.zero_grad()

            images = images.to(device)

            # xt, xt_minus_one, rand_steps = ddpm(images) # xt-1 is target given xt
            xt, noise_t, xt_minus_one, noise_t_minus_one, rand_steps = ddpm(images)
            
            xt, noise_t, rand_steps = xt.to(device), noise_t.to(device), rand_steps.to(device)
            pred_target = ddpm.model(xt, rand_steps)
            # pred_target is the "noise" predicted by the model; need this to be compared with "noise_t"

            loss = ddpm.criterion(pred_target, noise_t)
            # print(loss.item())
            losses.append(loss.item())
            loss.backward()
            ddpm.opt.step()

            if i % 100 == 0:
                print(loss.item(), i, ep)
        
        print(f"avg loss for epoch {ep} => {sum(losses)/len(losses)}")
        if (ep + 1) % 2 == 0:
            ddpm.reverse(ep, rev_batch_size = batch_size)
        
            save_path = os.path.join(save_dir, f"filtered_ep_{ep}.pt")
            torch.save(ddpm.state_dict(), save_path)
            print(f"saved model to {save_path}")

        print()
