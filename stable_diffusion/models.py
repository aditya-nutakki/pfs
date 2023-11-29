import torch, torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from  torch.optim.adam import Adam
from helpers import *
from time import time
import torchshow as ts



device = "cuda"

class LDM(nn.Module):
    def __init__(self, image_dims, beta1 = 10e-4, beta2 = 0.02, t = 1000, device = device) -> None:
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = t
    
        self.betas = torch.linspace(self.beta1, self.beta2, self.t)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim = -1)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.betas_hat = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


        self.image_dims = image_dims
        self.c, self.h, self.w = image_dims

        self.batch_size = 4 # used only for inference


        self.model = UNet(input_channels = self.c, output_channels = self.c)
        self.criterion = nn.MSELoss()
        self.opt = Adam(self.model.parameters(), lr = 2e-4)

        self.save_interval = 128

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
        batch_size = images.shape[0]
        rand_steps = torch.randint(1, self.t, (batch_size, ))
        (xt, noise_t), (xt_minus_one, noise_t_minus_one) = self.apply_noise(images, rand_steps), self.apply_noise(images, rand_steps-1) # need to predict xt_minus_one given xt

        # visualising
        # print(f"rand step => {rand_step}")
        # ts.save(xt, "xts.jpeg")
        # ts.save(xt_minus_one, "xt-1s.jpeg")
        return xt, noise_t, xt_minus_one, noise_t_minus_one, rand_steps


    def reverse(self, ep):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(self.batch_size, *self.image_dims, device = device)
            # denoise image here
            print("Denoising ...")
            stime = time()
            for t in range(self.t - 1, -1, -1):
                
                noise_comp = torch.randn(self.batch_size, *self.image_dims, device=device)*torch.sqrt(self.betas_hat[t]) # can multiply it with torch.sqrt(self.betas[t]) as well and you would get similar results
                one_by_alpha = (1/torch.sqrt(self.alphas[t]))
                _t = torch.Tensor([t]).type(torch.LongTensor).to(device)
                x = (one_by_alpha * (x - ((self.betas[t]) * self.model(x, _t) )/(torch.sqrt(1 - self.alphas_cumprod[t])))) + noise_comp

                if t % self.save_interval == 0:
                    ts.save(x, f"diffusion_{t}_{ep}.jpeg")
            ftime = time()

            print(f"reverse diffusion done in {ftime-stime}s for {self.t} time steps")




if __name__ == "__main__":
    ldm = LDM()


