from model import *
import torchshow as ts
import torch
from torchvision.transforms import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam


device = "cuda"



def eval(ep):
    model.eval()
    print("Evaluating ...")
    with torch.no_grad():
        for i, images in enumerate(loader):
            images = images.to(device)
            mean, log_var, preds = model(images)
            ts.save(preds, f"./eval_{ep}_{i}.jpg")
            print("saved")

            if i == 3:
                break
    


def train(loader):
    
    criterion = nn.MSELoss()
    lr = 3e-4
    opt = Adam(model.parameters(), lr = lr)
    print("Starting To Train ...")
    for ep in range(15):
        losses = []
        model.train()
        for i, images in enumerate(loader):

            opt.zero_grad()

            images = images.to(device)
            mean, log_var, preds = model(images)
            loss = criterion(preds, images)
            kl_loss = vae_gaussian_kl_loss(mean, log_var)
            # print(loss, kl_loss)
            loss = 5*loss + kl_loss
            loss.backward()

            opt.step()
            losses.append(loss.item())
            if i % 100 == 0:
                print(f"Loss on step {i}; epoch {ep} => {loss}")

        eval(ep)
        print(f"Avg loss for epoch {ep} => {sum(losses)/len(losses)}")
        print()


def get_loaders():
    ds = CLIPDataset("/mnt/d/work/datasets/faces/Humans")
    return DataLoader(ds, batch_size = 32, shuffle = True, num_workers = 2)



if __name__ == "__main__":
    loader = get_loaders()
    model = VAE()
    model = model.to(device)
    train(loader)





