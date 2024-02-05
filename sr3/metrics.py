import torch.nn as nn
import torch, torchvision
import torch.nn.functional as F
import os, numpy, json

import torchshow
from time import time

from helpers import *
from config import *
from diffusion import DiffusionModel
import math

from PIL import Image 
import torchvision.transforms as transforms 

def compute_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False):
    # (img1, img2) can be used interchangebly

    pad = window_size // 2
    
    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, height, width) # window should be atleast 11x11 
        window = create_window(real_size, channel=channels).to(img1.device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability 
    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean() 
    else: 
        ret = ssim_score.mean(1).mean(1).mean(1)
    
    if full:
        return ret, contrast_metric
    
    # outputs in the [0, 1] range
    return ret


def compute_fid():
    """
    requires 2 paths of generated images as well as real images.
    think through on how many images you want to pass it through

    """
    pass


def compute_psnr(x, y):
    # x, y to be of single shapes (c, h, w)
    
    psnr = lambda x, y: torch.log10(1.0/torch.sqrt(F.mse_loss(x,y))) * 20
    score = psnr(x, y)
    assert score >= 0

    return score


"""
Steps:
    1. evaluate on test set for different time steps (100, 200, 500, 1000, 1500)
        * sampling on all/most timesteps is very time taking on 20k images. 
    
    2. generate images with different time steps along with their corresponding hr image
    3. compute metrics wrt hr image and generated image.
    4. *current model needs to train for a lot longer. and see if we need to also train on 1000 timesteps


"""
    
def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def compute_metrics(generated_path, reference_path):
    # load images from both the paths, read them, convert them to tensors, pass them through individual metric computation blocks

    generated_images = os.listdir(generated_path)
    image2tensor = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, ), (0.5,))
    ])

    ssims, psnrs = [], []


    for i, generated_image_path in enumerate(generated_images):
        print(generated_image_path)
        generated_image = Image.open(os.path.join(generated_path, generated_image_path))
        # reference_image = Image.open(os.path.join(reference_path, generated_image_path[:-5]))
        reference_image = Image.open(os.path.join(reference_path, generated_image_path))
        
        generated_image, reference_image = image2tensor(generated_image), image2tensor(reference_image)

        
        print(generated_image.shape)
        ssim_score = compute_ssim(generated_image, reference_image)
        print(f"ssim score: {ssim_score}")

        # psnr_score = compute_psnr(generated_image, reference_image)
        # print(f"psnr score: {psnr_score}")
        
        print()

        ssims.append(ssim_score)
        # psnrs.append(psnr_score)
    
    print("Done !")
    print(f"Avg SSIM score: {sum(ssims)/len(ssims)}")
    # print(f"Avg PSNR score: {sum(psnrs)/len(psnrs)}")
    


"""
1. use 'dataloader' to test on test_set and save them accordingly
2. pass the paths as parameters to each of those metrics (ssim, fid, psnr)
3. 
"""


def generate_images(model, save_path = "/mnt/d/work/datasets/celebA_gen"):
    os.makedirs(save_path, exist_ok = True)
    loader = get_dataloader(path = "/mnt/d/work/datasets/celebA_test", hr_sz = hr_sz, lr_sz = lr_sz, batch_size = batch_size, return_img_path = True)
    
    ts = [500, 1000]

    for _t in ts:
        _t_path = os.path.join(save_path, str(_t))
        print(_t_path, os.path.exists(_t_path))
        os.makedirs(_t_path, exist_ok = True)

    print("Starting ...")
    for i, (hr_img, lr_img, img_name) in enumerate(loader):
        for _t in ts:
            _t_path = os.path.join(save_path, str(_t))
            
            if _t != model.time_steps:
                x = model.ddim_sample(lr_img, sample_steps = _t, eta = 1.0)
            else:
                x = model.sample(lr_img)
            # torchshow.save(x, f"{_t}ts.jpeg")
            save_images(x, save_path = _t_path, title = img_name)
            print(f"Done for t = {_t} timesteps")    
        
        print()
        if i == 208:
            break







if __name__ == "__main__":
    # x = torch.randn(3, 16, 16)
    # y = torch.randn(3, 16, 16)
    # y = x.clone()

    # x = torch.randn(3, 16, 16)
    # y = torch.ones(3, 16, 16)

    # print(compute_ssim(x, y))
    # compute_metrics(generated_path = "/mnt/d/work/datasets/celebA_gen/10", reference_path = "/mnt/d/work/datasets/celebA_test")
    
    ddpm = DiffusionModel(time_steps = time_steps)
    ddpm.load_state_dict(torch.load("./celeba_models/celeba_sr_ep_145_16x128_t2000.pt"))
    ddpm.to(device = "cuda")
    generate_images(ddpm)
    # for t = 10, ssim = 0.3253
    # for t = 50, ssim = 0.3120


