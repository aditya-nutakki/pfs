from model import *
import torchshow as ts
import torch
from torchvision.transforms import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch import Tensor
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.mnist import MNIST


class VanillaVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims = None,
                 kld_weight = 1.0,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        self.num_channels = in_channels
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            # hidden_dims = [32, 64, 128, 256, 512, 512]
            hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)
        
        self.kld_weight = kld_weight

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= self.num_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # print(f"init encoder shape => {result.shape}")
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        # print(f"enc op => {z.shape}")
        result = self.decoder_input(z)
        # print(f"init decoder result = {result.shape}")
        result = result.view(-1, 256, 2, 2)
        # print(f"view decoder result = {result.shape}")
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        recons_loss =F.mse_loss(recons, input, reduction="sum")
        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + self.kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)
    

device = "cuda"

def eval(ep, num_samples = 32):
    model.eval()
    print("Evaluating ...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            preds, input, mu, log_var = model(images)
            ts.save(preds, f"./recons_{ep}_{i}.jpg")
            break    
        
        sampled_images = model.sample(num_samples, current_device="cuda")
        ts.save(sampled_images, f"./sample_{ep}.jpg")
        
    print("saved")



def train():
    
    lr = 3e-4
    opt = Adam(model.parameters(), lr = lr)

    print("Starting...")

    for ep in range(20):
        model.train()
        losses = []

        for i, (images, labels) in enumerate(loader):
            opt.zero_grad()

            images = images.to(device)
            # ts.save(images, "./input.jpg")
            # exit()

            preds, input, mu, log_var = model(images)
            # print(f"preds shsape = {preds.shape}")
            loss = model.loss_function(preds, input, mu, log_var)
            loss, recon_loss, kl_loss = loss["loss"], loss["Reconstruction_Loss"], loss["KLD"]
            
            # print(loss.item(), recon_loss.item(), kl_loss.item())

            losses.append(loss.item())
            if i % 100 == 0:
                print(f"Loss on step {i}; epoch {ep} => {loss.item(), recon_loss.item(), kl_loss.item()}")

            loss.backward()
            opt.step()

        
        eval(ep)
        print(f"Avg loss for epoch {ep} => {sum(losses)/len(losses)}")
        print()


num_input_channels = 1
# model = VanillaVAE(in_channels = 3, latent_dim = 24, hidden_dims=[16, 32, 64, 32, 32])
model = VanillaVAE(in_channels = num_input_channels, latent_dim = 128, kld_weight = 0.8)
model = model.to(device)

# loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)

img_sz = 32

dataset = MNIST(root="./", download=True,
                        transform=transforms.Compose([
                        transforms.Resize((img_sz, img_sz)), # or h
                        transforms.ToTensor()
                        ]))

# dataset = CIFAR10(root="./", download=True,
#                         transform=transforms.Compose([
#                         transforms.Resize((img_sz, img_sz)), # or h
#                         transforms.ToTensor()
#                         ]))

loader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=2)

# loader = CLIPDataset("/mnt/d/work/datasets/faces/Humans", img_sz = img_sz)
# loader = DataLoader(loader, batch_size = 32, shuffle=True, num_workers=2)
train()

