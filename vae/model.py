import torch, torchvision
import torch.nn.functional as F
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os, cv2
from random import choice
from torch.distributions.normal import Normal



def vae_sample(mean, log_var):
    bs, dim = mean.shape        
    epsilon = Normal(0, 1).sample((bs, dim)).to(device="cuda")
    return mean + torch.exp(0.5 * log_var) * epsilon


class VAE_Encoder(nn.Module):
    def __init__(self, input_shape = (3, 224, 224), output_shape = (3, 16, 16), latent_dims = 16) -> None:
        super().__init__()
        self.c, self.h, self.w = input_shape
        self.latent_dims = latent_dims
        self.k = 24

        self.model = nn.Sequential(
            nn.Conv2d(in_channels = self.c, out_channels = 32, kernel_size = 5),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 5),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.mean_layer, self.log_var_layer = nn.Linear(3*self.k*self.k, self.latent_dims), nn.Linear(3*self.k*self.k, self.latent_dims)


    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        mean, log_var = self.mean_layer(x), self.log_var_layer(x)
        z = vae_sample(mean, log_var)
        # print(f"latent space dims => {z.shape}")
        return mean, log_var, z



class conv_block(nn.Module):
    def __init__(self, in_c, out_c, activation = "relu"):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()
        self.act = nn.ReLU() if activation == "relu" else nn.SiLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        # x = self.relu(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.relu(x)
        x = self.act(x)

        return x
    
    
class VAE_Decoder(nn.Module):
    def __init__(self, input_shape = (3, 24, 24), output_shape = (3, 224, 224), latent_dims = 16) -> None:
        super().__init__()
        self.k = 24
        self.conv1 = conv_block(64, 64)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 3, out_channels = 64, kernel_size = 2, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 3, kernel_size = 2, stride = 2),
            nn.ReLU(),
            # conv_block(32, 32),
            # nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 2, stride = 2),
        )
        self.input_shape = input_shape
        # self.latent_linear = nn.Linear(latent_dims, 3 * self.k * self.k)
        self.latent_linear = nn.Linear(latent_dims, 3 * 28 * 28)
        

    def forward(self, x):
        x = self.latent_linear(x)
        # x = x.view(-1, *self.input_shape)
        x = x.view(-1, 3, 28, 28)
        # print(f"init x => {x.shape}")
        x = self.model(x)
        # print(x.shape, x.mean().item(), x.std().item())
        x = torch.tanh(x)
        # print(x.shape, x.mean().item(), x.std().item())
        # print()
        return x


class VAE(nn.Module):
    def __init__(self, latent_dims = 32) -> None:
        super().__init__()
        self.latent_dims = latent_dims
        self.encoder = VAE_Encoder(latent_dims=self.latent_dims)
        self.decoder = VAE_Decoder(latent_dims=self.latent_dims)
        

    def forward(self, x):
        mean, log_var, z = self.encoder(x)
        return mean, log_var, self.decoder(z)


def vae_gaussian_kl_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return KLD.mean()


def reconstruction_loss(x_reconstructed, x):
    bce_loss = nn.BCELoss()
    return bce_loss(x_reconstructed, x)


def vae_loss(y_pred, y_true):
    mu, logvar, recon_x = y_pred
    recon_loss = reconstruction_loss(recon_x, y_true)
    kld_loss = vae_gaussian_kl_loss(mu, logvar)
    return 500 * recon_loss + kld_loss





# custom normalizing function to get into range you want
class NormalizeToRange(nn.Module):
    def  __init__(self, min_val, max_val) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, images):
        # images could be a batch or individual
        
        # _min_val, _max_val = torch.min(images), torch.max(images)
        # return (self.max_val - self.min_val) * ((images - _min_val) / (_max_val - _min_val)) + self.min_val
        return (self.max_val - self.min_val) * ((images - 0) / (1)) + self.min_val


class CLIPDataset(Dataset):
    def __init__(self, dataset_path, _transforms = None, max_len = 32, img_sz = 224) -> None:
        super().__init__()
        
        self.max_len = max_len
        self.transforms = _transforms
        
        if not self.transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_sz, img_sz)),
                transforms.Normalize((0.5, ), (0.5,))
            ])


        self.dataset_path = dataset_path
        # self.images_path, self.captions_path = os.path.join(self.dataset_path, "images"), os.path.join(self.dataset_path, "captions")
        # self.images, self.captions = os.listdir(self.images_path), os.listdir(self.captions_path)

        self.limit = 1500
        self.images_path = dataset_path
        self.images = os.listdir(self.images_path)[:self.limit]
        self.images = [os.path.join(self.images_path, image) for image in self.images]
        # self.captions = [os.path.join(self.captions_path, caption) for caption in self.captions]


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # to return a pair of (images, text-caption)
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image.shape)
        image = self.transforms(image)

        return image


class BikesDataset(Dataset):
    def __init__(self, dataset_path, limit = -1, _transforms = None, max_len = 32, img_sz = 224) -> None:
        super().__init__()
        
        self.max_len = max_len
        self.transforms = _transforms
        
        if not self.transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_sz, img_sz)),
                # transforms.Normalize((0.5, ), (0.5,))
                NormalizeToRange(-1, 1)
            ])


        self.dataset_path = dataset_path
        # self.images_path, self.captions_path = os.path.join(self.dataset_path, "images"), os.path.join(self.dataset_path, "captions")
        # self.images, self.captions = os.listdir(self.images_path), os.listdir(self.captions_path)
        self.valid_extensions = ["jpg", "jpeg", "png", "JPEG", "JPG"]
        self.limit = limit
        self.images_path = dataset_path
        self.images = os.listdir(self.images_path)[:self.limit]

        self.images = [os.path.join(self.images_path, image) for image in self.images if image.split(".")[-1] in self.valid_extensions]
        # self.captions = [os.path.join(self.captions_path, caption) for caption in self.captions]
        # self.images = self.preprocess_images(self.images)
        # print(self.images[:20])


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # to return a pair of (images, text-caption)
        try:
            image = cv2.imread(self.images[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(image.shape)
            image = self.transforms(image)
            return image
        
        except:
            return None
        

    def preprocess_images(self, image_paths):
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




if __name__ == "__main__":
    # ds_path = "/mnt/d/work/datasets/coco_captions"
    # ds = CLIPDataset(ds_path)

    # unet = UNet(down_factor=4)
    # print(sum([p.numel() for p in unet.parameters()]))
    # x, t = torch.randn(4, 3, 224, 224), torch.Tensor([11]).type(torch.LongTensor)
    # y = unet(x, t)
    # print(y.shape)

    # enc = VAE_Encoder()
    # print(sum([p.numel() for p in enc.parameters()]))
    # x = torch.randn(4, 3, 224, 224)
    # y = enc(x)
    # print(y.shape, y.mean(), y.std())

    latent_dims = 64
    enc = VAE_Decoder(latent_dims=latent_dims)
    print(sum([p.numel() for p in enc.parameters()]))
    x = torch.randn(4, latent_dims)
    y = enc(x)
    print(y.shape, y.mean(), y.std())

    # vae = VAE()
    # print(sum([p.numel() for p in vae.parameters()]))
    # x = torch.randn(4, 3, 224, 224)
    # y = vae(x)
    # print(y.shape)
    # pass