

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchshow as ts

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = './'
BATCH_SIZE = 128
IMAGE_CHANNEL = 1
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 16
EPOCH_NUM = 5
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
seed = 1


device = "cuda"

# cudnn.benchmark = True


# Data preprocessing
dataset = dset.MNIST(root="./", download=False,
                     transform=transforms.Compose([
                     transforms.Resize(X_DIM),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,))
                     ]))

# Dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=2)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()

# fixed noise
viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")


for ep in range(5):
    for i, (images, labels) in enumerate(dataloader):
        optimizerD.zero_grad()
        
        images = images.to(device)

        bs = list(images.shape)[0]
        real_labels, fake_labels = torch.ones(bs).to(device), torch.zeros(bs).to(device)

        noise = torch.randn(bs, Z_DIM, 1, 1).to(device)
        gen_preds = netG(noise)
        
        disc_preds_gt = netD(images)
        disc_real_loss = criterion(disc_preds_gt, real_labels)
        disc_real_loss.backward()

        disc_preds_fake = netD(gen_preds.detach())
        disc_fake_loss = criterion(disc_preds_fake, fake_labels)
        disc_fake_loss.backward()

        total_disc_loss = disc_fake_loss + disc_real_loss
        optimizerD.step()
        

        # generator
        optimizerG.zero_grad()
        # real_labels = torch.ones(bs).to(device)
        disc_preds_fake = netD(gen_preds)
        gen_loss = criterion(disc_preds_fake, real_labels)

        gen_loss.backward()
        optimizerG.step()

        


        if (i % 250 == 0) or ((ep == EPOCH_NUM-1) and (i == len(dataloader)-1)):
            print(f"disc_loss {total_disc_loss}; gen_loss {gen_loss}; epoch {ep}_{i}; disc_rl {disc_real_loss}; disc_fl {disc_fake_loss}")
            with torch.no_grad():
                fake = netG(viz_noise).detach().cpu()
                ts.save(fake, f"sample_{ep}_{i}.jpeg")











################################# WORKING ###########################


# for epoch in range(EPOCH_NUM):
#     for i, data in enumerate(dataloader, 0):

#         # (1) Update the discriminator with real data
#         # netD.zero_grad()
#         optimizerD.zero_grad()
#         real_cpu = data[0].to(device)
#         b_size = real_cpu.size(0)
#         label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
#         # Forward pass real batch through D
#         output = netD(real_cpu).view(-1)
#         # Calculate loss on all-real batch
#         errD_real = criterion(output, label)
#         errD_real.backward()
#         D_x = output.mean().item()

#         # (2) Update the discriminator with fake data
#         # Generate batch of latent vectors
#         noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)
#         # Generate fake image batch with G
#         fake = netG(noise)
#         label.fill_(FAKE_LABEL)
#         # Classify all fake batch with D
#         output = netD(fake.detach()).view(-1)
#         # Calculate D's loss on the all-fake batch
#         errD_fake = criterion(output, label)
#         # Calculate the gradients for this batch, accumulated (summed) with previous gradients
#         errD_fake.backward()
#         D_G_z1 = output.mean().item()
#         # Compute error of D as sum over the fake and the real batches
#         errD = errD_real + errD_fake
#         # Update D
#         optimizerD.step()

#         # (3) Update the generator with fake data
#         # netG.zero_grad()
#         optimizerG.zero_grad()
#         label.fill_(REAL_LABEL)  # fake labels are real for generator cost
#         # Since we just updated D, perform another forward pass of all-fake batch through D
#         output = netD(fake).view(-1)
#         # Calculate G's loss based on this output
#         errG = criterion(output, label)
#         # Calculate gradients for G
#         errG.backward()
#         D_G_z2 = output.mean().item()
#         # Update G
#         optimizerG.step()

#         # Output training stats
#         if i % 50 == 0:
#             print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
#                   % (epoch, EPOCH_NUM, i, len(dataloader),
#                      errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

#         # Save Losses for plotting later
#         G_losses.append(errG.item())
#         D_losses.append(errD.item())

#         # Check how the generator is doing by saving G's output on fixed_noise
        # if (iters % 500 == 0) or ((epoch == EPOCH_NUM-1) and (i == len(dataloader)-1)):
        #     with torch.no_grad():
        #         fake = netG(viz_noise).detach().cpu()
        #         ts.save(fake, f"sample_{epoch}_{i}.jpeg")
            # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

#         iters += 1

