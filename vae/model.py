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

    
class VAE_Decoder(nn.Module):
    def __init__(self, input_shape = (3, 24, 24), output_shape = (3, 224, 224), latent_dims = 16) -> None:
        super().__init__()
        self.k = 24
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 128, kernel_size = 7, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 128, out_channels = 3, kernel_size = 7, stride = 2, padding = 2, output_padding=1),
        )
        self.input_shape = input_shape
        self.latent_linear = nn.Linear(latent_dims, 3 * self.k * self.k)
        

    def forward(self, x):
        x = self.latent_linear(x)
        x = x.view(-1, *self.input_shape)
        x = self.model(x)
        # print(x.shape)
        return torch.tanh(x)


class VAE(nn.Module):
    def __init__(self, latent_dims = 16) -> None:
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



class CLIPDataset(Dataset):
    def __init__(self, dataset_path, _transforms = None, max_len = 32) -> None:
        super().__init__()
        
        self.max_len = max_len
        self.transforms = _transforms
        
        if not self.transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5, ), (0.5,))
            ])

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # print(type(self.tokenizer))

        self.dataset_path = dataset_path
        # self.images_path, self.captions_path = os.path.join(self.dataset_path, "images"), os.path.join(self.dataset_path, "captions")
        # self.images, self.captions = os.listdir(self.images_path), os.listdir(self.captions_path)

        self.images_path = dataset_path
        self.images = os.listdir(self.images_path)
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

        # tokenize text
        # caption_path = self.captions[index]
        # # print(caption_path)
        # with open(caption_path) as f:
        #     captions = f.readlines()
        #     captions = [caption.strip() for caption in captions]
        #     caption = choice(captions)
        #     # caption = captions[0]
        #     # print(f"caption is: {caption}")
        #     caption = self.tokenizer(caption, return_tensors = "pt", max_length = self.max_len, padding = "max_length")
        #     caption["input_ids"] = caption["input_ids"].view(-1)
        #     # print(caption)
        # # print(self.tokenizer.batch_decode(caption["input_ids"], skip_special_tokens = True)) # -> to decode a tensor to list of decoded sentences

        # return image, caption
    
        return image






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

    # enc = VAE_Decoder()
    # print(sum([p.numel() for p in enc.parameters()]))
    # x = torch.randn(4, 16)
    # y = enc(x)
    # print(y.shape, y.mean(), y.std())

    vae = VAE()
    print(sum([p.numel() for p in vae.parameters()]))
    x = torch.randn(4, 3, 224, 224)
    y = vae(x)
    print(y.shape)
    pass