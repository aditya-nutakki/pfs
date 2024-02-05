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


"""
Algorithm:

1. Must have 6 channels as input (since hr_img and lr_img are concatenated) and output to have 3 channels
2. hr_img is to be noised and denoised, lr_img is present only for conditioning
3. Dataloader must return 2 images, hr image and lr image scaled to hr image dims (if our hr image is of dims 128x128 and lr image is of dims 32x32, then lr image is to be scaled up to 128x128)
4. Instead of passing in time as a condition, you must pass 'gamma' which is the same as 'alpha_t_hat' in the original ddpm paper
5. gamma is processed by using another mlp.

"""






os.makedirs(model_save_dir, exist_ok = True)
os.makedirs(img_save_dir, exist_ok = True)


class DiffusionModel(nn.Module):
    def __init__(self, time_steps, 
                 beta_start = 10e-4, 
                 beta_end = 0.02,
                 image_dims = image_dims,
                 output_channels = 3):
        
        super().__init__()
        self.time_steps = time_steps
        print(f"LOADING WITH {self.time_steps} STEPS DIFFUSION")
        self.image_dims = image_dims
        c, h, w = self.image_dims
        self.img_size, self.input_channels = h, c
        

        self.betas = torch.linspace(beta_start, beta_end, self.time_steps)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim = -1)

        self.model = UNet(input_channels =  2*c, output_channels = output_channels, time_steps = self.time_steps, down_factor = 1)


    def ddim_sample(self, lr_img, sample_steps = 500, eta = 0.0, title = None):
        assert sample_steps <= self.time_steps, f"sampling steps should be lesser than number of time steps"
        
        # lr_img to be a 4D tensor

        num_samples = lr_img.shape[0] 
        self.model.eval()
        print(f"sampling {num_samples} examples with ddim sampling ... ")

        with torch.no_grad():
            times = torch.linspace(1, self.time_steps - 1, sample_steps).to(torch.long)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))
            
            
            x = torch.randn(num_samples, self.input_channels, self.img_size, self.img_size, device = device)
            lr_img = lr_img.to(device)
            stime = time()
            # print(x.shape)
            for t, t_minus_one in time_pairs:
                # noise = torch.randn(num_samples, *self.latent_image_dims, device = device)
                noise = torch.randn(num_samples, self.input_channels, self.img_size, self.img_size, device = device)
                alpha_t, alpha_t_minus_one = self.alpha_hats[t], self.alpha_hats[t_minus_one]
                
                t = torch.tensor(t, device = device).long()
                # pred_noise = self.model(torch.cat([x, lr_img], dim = 1).to(device), alpha_t.to(device))
                pred_noise = self.model(torch.cat([x, lr_img], dim = 1).to(device), alpha_t.view(-1).to(device))
                sigma = eta * torch.sqrt((1-alpha_t_minus_one)/(1 - alpha_t) * (1 - (alpha_t/alpha_t_minus_one)))
                
                k = torch.sqrt(1 - alpha_t_minus_one - sigma**2)
                pred_x0 = torch.sqrt(alpha_t_minus_one) * (x - torch.sqrt(1 - alpha_t)*pred_noise)/torch.sqrt(alpha_t)

                x = pred_x0 + k * pred_noise + sigma * noise

            ftime = time()
            print(f"Done denoising in {ftime - stime}s ")
            # torchshow.save(x, os.path.join(img_save_dir, f"./sr_ddim_sample_{sample_steps}_18.jpeg"))
        return x


    def sample(self, lr_img):
        # reverse process
        self.model.eval()
        
        num_samples = lr_img.shape[0]
        print(f"Normally sampling {num_samples}")
        stime = time()
        with torch.no_grad():
        
            x = torch.randn(num_samples, self.input_channels, self.img_size, self.img_size).to(device)
            lr_img = lr_img.to(device)

            for i, t in enumerate(range(self.time_steps - 1, 0 , -1)):
                alpha_t, alpha_t_hat, beta_t = self.alphas[t], self.alpha_hats[t], self.betas[t]
                # print(alpha_t, alpha_t_hat, beta_t)
                t = torch.tensor(t, device = device).long()
                pred_noise = self.model(torch.cat([x, lr_img], dim = 1).to(device), alpha_t_hat.view(-1).to(device))
                # pred_noise = self.model(torch.cat([lr_img, x], dim = 1).to(device), alpha_t_hat.view(-1).to(device))
                x = (torch.sqrt(1/alpha_t))*(x - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * pred_noise)

                if i > 1:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta_t) * noise
                
        ftime = time()
        # torchshow.save(x, os.path.join(img_save_dir, f"sr_sample.jpeg"))
        print(f"Done denoising in {ftime - stime}s ")
        return x


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

    loader = get_dataloader(hr_sz = hr_sz, lr_sz = lr_sz, batch_size = batch_size)

    opt = torch.optim.Adam(ddpm.model.parameters(), lr = lr)
    criterion = nn.MSELoss(reduction="mean")

    # ddpm.load_state_dict(torch.load("./celeba_models/celeba_sr_ep_9_32x128_t2000.pt"))
    ddpm.load_state_dict(torch.load("./celeba_models/celeba_sr_ep_143_16x128_t2000.pt"))
    
    ddpm.model.to(device)
    offset = 144
    for ep in range(offset, epochs + offset):
        ddpm.model.train()
        print(f"Epoch {ep}:")
        losses = []
        stime = time()
        
        # for i, x in enumerate(loader):
        for i, (x, y) in enumerate(loader):
            
            # x is highres target image, y is low res image to which is to be conditioned
            
            bs = x.shape[0]
            x, y = x.to(device), y.to(device)
            

            ts = torch.randint(low = 1, high = ddpm.time_steps, size = (bs, ))
            gamma = ddpm.alpha_hats[ts].to(device)
            ts = ts.to(device = device)

            x, target_noise = ddpm.add_noise(x, ts)
            x = torch.cat([x, y], dim = 1)
            # print(x.shape, target_noise.shape)
            # print(x.shape)
            predicted_noise = ddpm.model(x, gamma)
            loss = criterion(target_noise, predicted_noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())

            if i % 250 == 0:
                print(f"Loss: {loss.item()}; step {i}; epoch {ep}")


        ftime = time()
        print(f"Epoch trained in {ftime - stime}s; Avg loss => {sum(losses)/len(losses)}")

        if ep % 1 == 0:
            # ddpm.sample(ep)
            torch.save(ddpm.state_dict(), os.path.join(model_save_dir, f"celeba_sr_ep_{ep}_{lr_sz}x{hr_sz}_t{time_steps}.pt"))
        print()
            

def eval(num_samples = batch_size):
    ddpm = DiffusionModel(time_steps = time_steps)
    # ddpm.load_state_dict(torch.load(os.path.join(model_save_dir, "sr_ep_7_64x128.pt")))
    ddpm.load_state_dict(torch.load(os.path.join(model_save_dir, "sr_ep_2_32x128.pt")))
    ddpm = ddpm.to(device)
    c, h, w = image_dims
    assert h == w, f"height and width must be same, got {h} as height and {w} as width"
    print(f"Loaded model, trying to sample !")
    
    loader = get_dataloader(dataset_type="sr", img_sz = h, batch_size = num_samples)

    for i, (hr_img, lr_img) in enumerate(loader):
        ddpm.sample(lr_img, num_samples=num_samples)
        print(f"done with normal sampling ...")
        print()
        ddpm.ddim_sample(lr_img, sample_steps = 100, num_samples = num_samples, eta = 1.0)

        torchshow.save(hr_img, os.path.join(img_save_dir, f"hr_img.jpeg"))
        torchshow.save(lr_img, os.path.join(img_save_dir, f"lr_img.jpeg"))

        break



if __name__ == "__main__":
    time_steps = 2000
    train_ddpm(time_steps = time_steps)
    # eval(4)

    # timesteps = 100
    # beta_start, beta_end = 10e-4, 0.02
    # betas = torch.linspace(beta_start, beta_end, timesteps)
    # alphas = 1 - betas
    # alpha_hats = torch.cumprod(alphas, dim = -1)
    
    # print(f"alpha hats: {alpha_hats}")
    # print()

    # ts = torch.randint(timesteps, (batch_size, ))
    # # gamma = torch.randn((batch_size, ))
    # gamma = alpha_hats[ts]
    # print(ts, gamma)

    # ddpm = DiffusionModel(time_steps)
    # ddpm.to(device)
    # ddpm.sample(4, 8)
