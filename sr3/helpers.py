import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, cv2
from torchvision.transforms import transforms

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10

from config import *
import matplotlib.pyplot as plt
from random import uniform

import torchshow
from torchvision.transforms import InterpolationMode
import math
from torchvision.utils import save_image

from PIL import Image


# custom normalizing function to get into range you want
class NormalizeToRange(nn.Module):
    def  __init__(self, min_val, max_val) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, images):
        # images could be a batch or individual
        
        _min_val, _max_val = torch.min(images), torch.max(images)
        return (self.max_val - self.min_val) * ((images - _min_val) / (_max_val - _min_val)) + self.min_val
        # return (self.max_val - self.min_val) * ((images - 0) / (1)) + self.min_val
    


os.makedirs(metrics_save_dir, exist_ok = True)    

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p = 0.2):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor):
        if uniform(0, 1) < self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor
    

augmentations_to_use = transforms.Compose([AddGaussianNoise(std = uniform(0.01, 0.1), p = 1)])


class BikesDataset(Dataset):
    def __init__(self, dataset_path, limit = -1, _transforms = None, img_sz = 128) -> None:
        super().__init__()
        
        self.transforms = _transforms
        
        if not self.transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_sz, img_sz)),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.ColorJitter([0.5, 1]),
                transforms.RandomAdjustSharpness(1.1, p = 0.4)
                # transforms.Normalize((0.5, ), (0.5,))
                # NormalizeToRange(-1, 1)
            ])


        self.dataset_path, self.limit = dataset_path, limit
        self.valid_extensions = ["jpg", "jpeg", "png", "JPEG", "JPG"]
        
        self.images_path = dataset_path
        self.images = os.listdir(self.images_path)[:self.limit]

        self.images = [os.path.join(self.images_path, image) for image in self.images if image.split(".")[-1] in self.valid_extensions] 
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)
        return image

    def preprocess_images(self, image_paths):
        # call this method incase you think there are invalid images
        clean_paths = []
        count = 0
        for index, image_path in enumerate(image_paths):
            try:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # print(image.shape)
                image = self.transforms(image)
                clean_paths.append(image_path)
            except:
                print(f"failed at {image_path}")
                count += 1

        print(f"{count} number of invalid images")
        return clean_paths



class SRDataset(Dataset):
    def __init__(self, dataset_path, limit = -1, _transforms = None, hr_sz = 128, lr_sz = 32, return_img_path = False) -> None:
        super().__init__()
        
        self.transforms = _transforms
        self.return_img_path = return_img_path
        
        if not self.transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Resize((hr_sz, hr_sz)),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.ColorJitter([0.5, 1]),
                transforms.RandomAdjustSharpness(1.1, p = 0.4),
                transforms.Normalize((0.5, ), (0.5,))
                # NormalizeToRange(-1, 1)
            ])

        self.hr_sz, self.lr_sz = transforms.Resize((hr_sz, hr_sz), interpolation=InterpolationMode.BICUBIC), transforms.Resize((lr_sz, lr_sz), interpolation=InterpolationMode.BICUBIC)
        
        self.dataset_path, self.limit = dataset_path, limit
        self.valid_extensions = ["jpg", "jpeg", "png", "JPEG", "JPG"]
        
        self.images_path = dataset_path
        self.images = os.listdir(self.images_path)[:self.limit]

        self.images = [os.path.join(self.images_path, image) for image in self.images if image.split(".")[-1] in self.valid_extensions] 
    

    def preprocess_images(self, image_paths):
        # call this method incase you think there are invalid images
        clean_paths = []
        count = 0
        for index, image_path in enumerate(image_paths):
            try:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # print(image.shape)
                image = self.transforms(image)

                clean_paths.append(image_path)
            except:
                print(f"failed at {image_path}")
                count += 1

        print(f"{count} number of invalid images")
        return clean_paths
    

    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)
        hr_image, lr_image = self.hr_sz(image), self.lr_sz(image)

        if self.return_img_path:
            return hr_image, self.hr_sz(lr_image), self.images[index].split("/")[-1]

        return hr_image, self.hr_sz(lr_image)



def get_dataloader(path = "/mnt/d/work/datasets/celebA", batch_size = 8, hr_sz = 128, lr_sz = 32, return_img_path = False,limit = -1):

    ds = SRDataset(path, hr_sz = hr_sz, lr_sz = lr_sz, return_img_path = return_img_path)
    
    print(f"Training on {len(ds)} samples; with batch size {batch_size}; image dims {image_dims}; hr_sz {hr_sz}; lr_sz {lr_sz} ...")
    return DataLoader(ds, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 4)


def plot_metrics(losses, title, save_path = None, x_label = "steps", y_label = "loss"):
    plt.plot(losses)
    # Adding labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save_path is None:
        save_path = os.path.join(metrics_save_dir, f"{title}.jpeg")
    plt.savefig(save_path)
    plt.close()


def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel = 1):

    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    
    # Converting to 2D  
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)     
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window



def save_images(images, save_path = "./", title = "sample"):
    print("saving images ...")
    ntr = NormalizeToRange(0, 1)
    for i, image in enumerate(images):
        # torchshow.save(image, os.path.join(save_path, f"{title[i]}.jpeg"))
        image = ntr(image)
        save_image(image, os.path.join(save_path, title[i]))




if __name__ == "__main__":
    ds = SRDataset("/mnt/d/work/datasets/nature/x128/all", hr_sz=128)
    hr_img, lr_img = ds[12]
    print(hr_img.std(), hr_img.mean())
    print(lr_img.std(), lr_img.mean())
    print()
    print(torch.min(hr_img), torch.max(hr_img))
    print(torch.min(lr_img), torch.max(lr_img))

    ts.save(hr_img, "./hrimg.jpeg")
    ts.save(lr_img, "./lrimg.jpeg")
