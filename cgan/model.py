import os
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchshow as ts


# image_dims = (3, 64, 64) # c, h, w
c, h, w = 3, 64, 64
image_dims = (c, h, w) # c, h, w
hidden_dims = 64
batch_size = 32
criterion = nn.BCELoss()
lr = 0.0002 
ndf = 8
# class Discriminator(nn.Module):
#     def __init__(self, image_dims = image_dims) -> None:
#         super().__init__()
#         self.image_dims = image_dims
#         self.c, self.h, self.w = self.image_dims

#         self.discriminator = nn.Sequential(
#             nn.Conv2d(self.c, 32, kernel_size=3),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.1),
#             nn.Flatten(start_dim=1),
#             nn.Linear(123008, 256),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.1),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.discriminator(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(c, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self, image_dims = image_dims, hidden_dims = hidden_dims) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.image_dims = image_dims
        self.c, self.h, self.w = self.image_dims
        self.target_size = self.c * self.h * self.w

        self.generator = nn.Sequential(
            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(self.c, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 512 x 4 x 4

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # out: 256 x 8 x 8

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # out: 128 x 16 x 16

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # out: 64 x 32 x 32

            nn.ConvTranspose2d(8, self.c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # out: 3 x 64 x 64
        )


    def forward(self, x):
        # x to be of the same target shape but noise in the range [-1, 1]. x to be of the shape (batch_size, *image_dims) output dims should match as well.
        return self.generator(x)
    

"""
Discriminator:
    1. pass real images with generated images with target labels == 1. this is assuming generator "always" produces "fake" images
    2. when passed real images, targets should be set to torch.ones(). 
    3. when generator's output is passed, targets should be set to torch.zeros().
    2. loss computed in step 1 should be minimized with discriminator's params

Generator:
    1. pass noise into generator model.
    2. generator model's output should be used as input for discriminator
    3. output from step 2 and torch.ones() should be used as loss inputs
    3. loss computed in step 3. should be reduced with the optimizer for generator's params 
    
"""

device = "cuda"
def train_disc(gen, disc, gt_images, disc_opt):
    
    gen, disc = gen.to(device), disc.to(device)
    
    disc_targets_for_generator, disc_targets_for_gt = torch.zeros((batch_size, 1)).to(device), torch.ones((batch_size, 1)).to(device)

    
    gen.eval()
    disc.train()
    disc_opt.zero_grad()

    gt_images = gt_images.to(device)

    gen_inputs = torch.randn(batch_size, *image_dims).to(device)
    # print(f"gen_inputs shape => {gen_inputs.shape}")
    gen_preds = gen(gen_inputs)
    # print(f"gen_preds shape => {gen_preds.shape}")
    disc_preds_for_generator, disc_preds_for_gt = disc(gen_preds), disc(gt_images)
    disc_loss_for_gt = criterion(disc_preds_for_gt, disc_targets_for_gt)
    disc_loss_for_generator = criterion(disc_preds_for_generator, disc_targets_for_generator)

    # how do we go about the backward pass now ? 
    loss = disc_loss_for_gt + disc_loss_for_generator
    loss.backward()

    disc_opt.step()

    return loss.item()


def train_gen(gen, disc, gen_opt):
    
    gen, disc = gen.to(device), disc.to(device)
    
    gen.train()
    disc.eval()
    
    gen_opt.zero_grad()
    
    gen_targets = torch.ones((batch_size, 1)).to(device)

    _noise = torch.randn(batch_size, *image_dims).to(device)
    gen_preds = gen(_noise)
    disc_outputs = disc(gen_preds)
    gen_loss = criterion(disc_outputs, gen_targets)
    
    gen_loss.backward()
    gen_opt.step()

    return gen_loss.item()


def train(gen, disc):
    train_loader, test_loader = get_mnist_loader()
    gen_opt = Adam(gen.parameters(), lr = lr)
    disc_opt = Adam(disc.parameters(), lr = lr)
    
    
    print("Starting to train ...")

    for e in range(10):
        gen.train()
        disc.train()
        disc_losses, gen_losses = [], []

        for i, (images, _) in enumerate(train_loader):
            # print(images.shape)
            disc_loss = train_disc(gen, disc, images, disc_opt)
            gen_loss = train_gen(gen, disc, gen_opt)
            disc_losses.append(disc_loss)
            gen_losses.append(gen_loss)



            if i % 100 == 0:
                print(f"disc_loss => {disc_loss}, gen_loss => {gen_loss}")
                
        
        print(f"Avg Disc Loss => {sum(disc_losses)/len(disc_losses)}; Avg Gen Loss => {sum(gen_losses)/len(gen_losses)}; in epoch => {e}")
        print()

        gen.eval()
        with torch.no_grad():
            _noise = torch.randn(4, *image_dims).to(device)
            preds = gen(_noise)
            ts.save(preds, f"sample_{e}.jpg")
            print(f"saved results for epoch {e}")



def get_mnist_loader():
    # cifar-10 in rgb format
    # mean, std = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618], [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

    # mnist 
    mean, std = [0.1307], [0.3081]

    _transforms = transforms.Compose(
        [transforms.Resize(image_dims[1]), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    train_set, test_set = CIFAR10("./", train=True, transform=_transforms, download=True), CIFAR10("./", train=False, transform=_transforms, download=True)
    # train_set, test_set = MNIST("./", train=True, transform=_transforms, download=True), MNIST("./", train=False, transform=_transforms, download=True)
    return DataLoader(train_set, batch_size = batch_size, shuffle=True), DataLoader(test_set, batch_size = batch_size, shuffle=True)

if __name__ == "__main__":
    x = torch.randn(batch_size, *image_dims)
    d = Discriminator()
    g = Generator()
    # y = d(x)
    # print(y, y.shape)
    train(g, d)
