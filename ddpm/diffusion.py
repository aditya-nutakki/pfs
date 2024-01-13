import torch.nn as nn
import torch, torchvision
import torch.nn.functional as F
import os, numpy, json
# from unet_time import UNet
from unet_attn import UNet
import torchshow
from time import time

from helpers import *
from config import *


os.makedirs(model_save_dir, exist_ok = True)
os.makedirs(img_save_dir, exist_ok = True)


class DiffusionModel(nn.Module):
    def __init__(self, time_steps, 
                 beta_start = 10e-4, 
                 beta_end = 0.02,
                 image_dims = image_dims,
                 output_channels = 1):
        
        super().__init__()
        self.time_steps = time_steps
        print(f"LOADING WITH {self.time_steps} STEPS DIFFUSION")
        self.image_dims = image_dims
        c, h, w = self.image_dims
        self.img_size, self.input_channels = h, c
        

        self.betas = torch.linspace(beta_start, beta_end, self.time_steps)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim = -1)

        self.model = UNet(input_channels = c, output_channels = output_channels, time_steps = self.time_steps, down_factor = 1)


    def sample(self, ep, num_samples = batch_size):
        # reverse process
        self.model.eval()

        print(f"Sampling {num_samples} samples...")
        stime = time()
        with torch.no_grad():
            x = torch.randn(num_samples, self.input_channels, self.img_size, self.img_size, device = device)
            for i, t in enumerate(range(self.time_steps - 1, 0 , -1)):
                alpha_t, alpha_t_hat, beta_t = self.alphas[t], self.alpha_hats[t], self.betas[t]
                # print(alpha_t, alpha_t_hat, beta_t)
                t = torch.tensor(t, device = device).long()
                x = (torch.sqrt(1/alpha_t))*(x - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * self.model(x, t))
                if i > 1:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta_t) * noise
        ftime = time()
        torchshow.save(x, os.path.join(img_save_dir, f"sample_{ep}.jpeg"))
        print(f"Done denoising in {ftime - stime}s ")


    def add_noise(self, x, ts):
        noise = torch.randn_like(x)
        # print(x.shape, noise.shape)
        noised_examples = []
        for i, t in enumerate(ts):
            alpha_hat_t = self.alpha_hats[t]
            noised_examples.append(torch.sqrt(alpha_hat_t)*x[i] + torch.sqrt(1 - alpha_hat_t)*noise[i])

        return torch.stack(noised_examples), noise
        

    def forward(self, x, t):
        return self.model(x, t)



def train_ddpm(time_steps = time_steps, epochs = epochs):
    ddpm = DiffusionModel(time_steps = time_steps)
    c, h, w = image_dims
    assert h == w, f"height and width must be same, got {h} as height and {w} as width"

    loader = get_dataloader(dataset_type="mnist", img_sz = h, batch_size = batch_size)

    opt = torch.optim.Adam(ddpm.model.parameters(), lr = lr)
    criterion = nn.MSELoss(reduction="mean")

    ddpm.model.to(device)
    for ep in range(epochs):
        ddpm.model.train()
        print(f"Epoch {ep}:")
        losses = []
        stime = time()
        
        # for i, x in enumerate(loader):
        for i, (x, _) in enumerate(loader):
            bs = x.shape[0]
            x = x.to(device)
            ts = torch.randint(low = 1, high = ddpm.time_steps, size = (bs, ), device = device)

            x, target_noise = ddpm.add_noise(x, ts)
            # print(x.shape, target_noise.shape)
            # print(x.shape)
            predicted_noise = ddpm.model(x, ts)
            loss = criterion(target_noise, predicted_noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())

            if i % 200 == 0:
                print(f"Loss: {loss.item()}; step {i}; epoch {ep}")

        ftime = time()
        print(f"Epoch trained in {ftime - stime}s; Avg loss => {sum(losses)/len(losses)}")

        if (ep + 1) % 1 == 0:
            ddpm.sample(ep)
        
        print()
            



if __name__ == "__main__":
    # time_steps = 1000
    train_ddpm(time_steps = time_steps)



