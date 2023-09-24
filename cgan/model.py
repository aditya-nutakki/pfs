import os
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

image_dims = (3, 64, 64) # c, h, w
hidden_dims = 64
batch_size = 8
criterion = nn.BCELoss()
lr = 3e-4

class Discriminator(nn.Module):
    def __init__(self, image_dims = image_dims) -> None:
        super().__init__()
        self.image_dims = image_dims
        self.c, self.h, self.w = self.image_dims

        self.discriminator = nn.Sequential(
            nn.Conv2d(self.c, 32, kernel_size=3),
            nn.LeakyReLU(0.1),
            nn.Flatten(start_dim=1),
            nn.Linear(123008, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)
    


class Generator(nn.Module):
    def __init__(self, image_dims = image_dims, hidden_dims = hidden_dims) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.image_dims = image_dims
        self.c, self.h, self.w = self.image_dims
        self.target_size = self.c * self.h * self.w

        self.generator = nn.Sequential(
            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(self.c, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 4 x 4

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 3 x 64 x 64
        )


    def forward(self, x):
        # x to be of the same target shape but noise in the range [-1, 1]. x to be of the shape (batch_size, *image_dims) output dims should match as well.
        return self.generator(x)
    

class GAN(nn.Module):
    def __init__(self, generator, discriminator) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        
        self.gen_opt, self.dis_opt = Adam(self.generator.parameters(), lr = lr), Adam(self.discriminator.parameters(), lr = lr)


    def train_gen(self):
        return 
    
    def train_disc(self):
        return

    def forward(self, x):
        return x

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
def train(gen, disc, train_loader):
    for e in range(10):
        gen.train()
        disc.train()

        for idx, (gt_images, gt_labels) in enumerate(train_loader):
            gt_images = gt_images.to(device)

            gen_inputs = torch.randn(batch_size, *image_dims)
            gen_preds = gen(gt_images)
            
            disc_preds = disc(gen_preds)

            



if __name__ == "__main__":
    x = torch.randn(batch_size, *image_dims)
    d = Discriminator()
    # g = Generator()

    y = d(x)
    print(y, y.shape)

