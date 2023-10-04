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
import numpy

import warnings
warnings.filterwarnings("ignore")


DATA_PATH = './'
BATCH_SIZE = 128
IMAGE_CHANNEL = 1
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 16
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
num_classes = 10
save_dir = "./"
device = "cuda"

EPOCH_NUM = 40
min_num, max_num = -1, 1

dataset = dset.MNIST(root="./", download=False,
                     transform=transforms.Compose([
                     transforms.Resize(X_DIM),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,))
                     ]))

# Dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)



def get_positional_encodings(labels, num_classes = num_classes, min_num = min_num, max_num = max_num):
    # labels is a tensor of shape (batch_size, ) containing labels ranging from [0, num_classes)
    # print(f"labels => {labels}")
    labels = min_num + ((labels - 0)/(num_classes - 1 - 0))*((max_num - min_num)) + torch.exp(labels/num_classes) # subtracting with 0 as the minimum value of labels is 0
    labels = labels.tolist()
    new_labels = []
    for i in range(len(labels)):
        temp = []
        for _ in range(num_classes):
            temp.append(labels[i])
        new_labels.append(temp)
    new_labels = np.array(new_labels).astype("float32")
    # print(f"\n new labels => {new_labels}")
    return torch.from_numpy(new_labels).to(device)




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(num_classes,  num_classes)
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(Z_DIM + num_classes, G_HIDDEN * 8, 4, 1, 0, bias=False),
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

    def get_pos_encoding(self, labels, k = 1.5):
        # 0.0 represents class 0; 0.9 represents class 9
        # return torch.ones((BATCH_SIZE, num_classes), device = device) * (1/10)
        # print(labels.shape, labels)
        return labels/(num_classes * k)


    def forward(self, input, labels = None):
        # class_embedding = self.embedding(labels).view(-1, num_classes, 1, 1)
        class_embedding = get_positional_encodings(labels).view(-1, num_classes, 1, 1)
        input = torch.cat([input, class_embedding], dim = 1) # is of shape (batch_size, channels, h, w)
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.flatten = nn.Flatten()
       
        self.model = nn.Sequential(
            nn.Linear((IMAGE_CHANNEL * X_DIM**2) + num_classes, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
            )
    
    def get_pos_encoding(self, labels, k=1.5):
        # 0.0 represents class 0; 0.9 represents class 9
        # return torch.ones((BATCH_SIZE, num_classes), device = device) * (1/10)
        return labels/(num_classes*k)

    def forward(self, input, labels = None):
        # labels = self.embedding(labels)
        labels = get_positional_encodings(labels).view(-1, num_classes, 1, 1)
        input, labels = self.flatten(input), self.flatten(labels)
        input = torch.cat([input, labels], dim = 1)
        # return self.main(input).view(-1)
        return self.model(input).view(-1)


netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()

# fixed noise
fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
fixed_labels = torch.randint(0, num_classes, (BATCH_SIZE, ), device=device)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


img_list = []
G_losses = []
D_losses = []
iters = 0


def save_generator(model, name = "generator"):
    ckpt = {}
    ckpt["model_state_dict"] = model.state_dict()
    ckpt["model"] = model

    torch.save(ckpt, os.path.join(save_dir, f"{name}.pt"))
    print(f"saved model to {save_dir}")


def train():
    print("Starting Training Loop...")
    for ep in range(EPOCH_NUM):
        for i, (images, labels) in enumerate(dataloader):
            optimizerD.zero_grad()
            
            images, labels = images.to(device), labels.to(device)

            bs = list(images.shape)[0] # batch_size
            real_labels, fake_labels = torch.ones(bs).to(device), torch.zeros(bs).to(device)

            noise = torch.randn(bs, Z_DIM, 1, 1, device=device)
            labels_for_generator = torch.randint(0, num_classes, (bs, ), device=device)
            
            gen_preds = netG(noise, labels_for_generator)
            
            disc_preds_gt = netD(images, labels) # real preds wrt labels
            disc_real_loss = criterion(disc_preds_gt, real_labels)
            disc_real_loss.backward()

            disc_preds_fake = netD(gen_preds.detach(), labels_for_generator)
            disc_fake_loss = criterion(disc_preds_fake, fake_labels)
            disc_fake_loss.backward()

            total_disc_loss = disc_fake_loss + disc_real_loss
            optimizerD.step()
            
            # generator
            optimizerG.zero_grad()
            disc_preds_fake = netD(gen_preds, labels_for_generator)
            gen_loss = criterion(disc_preds_fake, real_labels)

            gen_loss.backward()
            optimizerG.step()


            if (i % 300 == 0) or ((ep == EPOCH_NUM-1) and (i == len(dataloader)-1)):
                print(f"disc_loss {total_disc_loss}; gen_loss {gen_loss}; epoch {ep}_{i}; disc_rl {disc_real_loss}; disc_fl {disc_fake_loss}")
                with torch.no_grad():
                    fake = netG(fixed_noise, fixed_labels).detach().cpu()
                    ts.save(fake, f"sample_{ep}_{i}.jpeg")


    save_generator(netG, "generator_new")


if __name__ == "__main__":
    train()