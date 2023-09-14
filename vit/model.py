import numpy as np

from tqdm import tqdm, trange
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from math import sin, cos



class FeedForward(nn.Module):
    def __init__(self, d, dff) -> None:
        super().__init__()
        self.dff = dff
        self.d = d
        self.linear1 = nn.Linear(d, dff)
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(dff, d)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        # print(x.shape)
        x = self.dropout(x)
        return self.linear2(x)
    


class MultiHeadAttention(nn.Module):
    def __init__(self, d = 8, num_heads = 8, patch_dim = 7) -> None:
        super().__init__()
        self.d = d
        self.h = num_heads
        assert d % num_heads == 0, f"cant divide {d} hidden dimensions by {num_heads} heads"
        self.dk = self.d // num_heads
        self.dv = self.d // num_heads
        self.dropout = nn.Dropout(0.25)
        self.n_patch = patch_dim**2 + 1

        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        
        self.wo = nn.Linear(d, d)
        # print(f"patch_dim in msa => {patch_dim}")q
    def attention(self, q, k, v):
        dot_product = F.softmax((q @ k.transpose(-1, -2))/self.dk**0.5, dim = -1)
        return dot_product @ v, dot_product


    def forward(self, q, k, v):
        # q, k, v to be of the shape => (-1, 49 + 1, d_hidden) -> (-1, h, 49+1, dk)
        q = self.wq(q).view(-1, self.n_patch, self.h, self.dk).transpose(1, 2)
        k = self.wk(k).view(-1, self.n_patch, self.h, self.dk).transpose(1, 2)
        v = self.wv(v).view(-1, self.n_patch, self.h, self.dv).transpose(1, 2)
        
        attention, dot_product = self.attention(q, k, v)
        # attention to be of the shape (-1, h, self.n_patch, dk)
        attention = attention.transpose(1, 2).reshape(-1, self.n_patch, self.d) # d can also be written as h * self.dv
        # attention = attention.reshape(-1, self.n_patch, self.d) # this works too 
        return self.wo(attention), dot_product


class EncoderBlock(nn.Module):
    def __init__(self, d, num_heads, dff = 128, patch_dim = 7):
        super().__init__()
        
        self.d = d
        self.num_heads = num_heads
        self.dff = dff
        self.patch_dim = patch_dim

        self.layer_norm = nn.LayerNorm(self.d)
        # print(f"patch dim in encoder block => {self.patch_dim}")
        self.mha = MultiHeadAttention(self.d, self.num_heads, self.patch_dim)
        self.ff = FeedForward(self.d, self.dff)

    def forward(self, x):
        normal_x = self.layer_norm(x)
        _attention_output, dot_product = self.mha(normal_x, normal_x, normal_x)
        x = x + _attention_output
        x = x + self.ff(self.layer_norm(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, d, n_heads, n_layers, patch_dim = None) -> None:
        super().__init__()
        self.d = d
        self.n_layers = n_layers
        self.patch_dim = patch_dim
        self.n_heads = n_heads
        # print(f"patch dim in VitEncoder => {self.patch_dim}")
        self.encoders = nn.ModuleList([EncoderBlock(d = self.d, num_heads = self.n_heads, patch_dim = self.patch_dim) for _ in range(n_layers)])

    def forward(self, x):
        for e in self.encoders:
            x = e(x)
        return x


class ViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super().__init__()
        
        # Attributes
        self.chw = chw # c, h, w
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.d)
        
        self.class_token = nn.Parameter(torch.rand(1, self.d))
        
        self.blocks = nn.ModuleList([EncoderBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        self.mlp = nn.Linear(self.d, out_d)

        self.patch_dim = self.chw[0] * (self.patch_size[0]**2)
        
        self.pos_embeds = self.get_pos_embedding(n_patches**2 + 1, self.d)
        self.vit_encoders = ViTEncoder(d = self.d, n_heads= self.n_heads, n_layers= 4, patch_dim=self.n_patches)


    def get_pos_embedding(self, n, d):
        pos_embedding = torch.zeros(n, d)

        for i in range(n):
            for j in range(d):
                pos_embedding[i][j] = sin(i/ 10000**(j/d)) if j % 2 == 0 else cos(i/ 10000**(j-1/d))
                
        pos_embedding = nn.Parameter(pos_embedding)
        pos_embedding.requires_grad = False

        return pos_embedding


    def forward(self, x):
    
        x = x.view(-1, self.n_patches**2, self.patch_dim)
        x = self.linear_mapper(x)
        x = torch.stack(
            [torch.cat([self.class_token, _x], dim = 0) for _x in x] # we add the class_token embedding for each image coming in; we need to iterate over batch_size so thaat all of them get it
        )
        
        x += self.pos_embeds
        # you now have a tensor of shape (batch_size, n_patches**2 + 1, d); we must now normalise and then pass it through the MHA module -> this can be encapsulated in a single encoder block
        x = self.vit_encoders(x)
        x = x[:, 0]
        
        return F.softmax(self.mlp(x), dim = -1) 
    

def main():
    # Loading data
    transform = ToTensor()

    train_set = CIFAR10(root='./', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    
    model = ViT((3, 32, 32), n_patches=8, n_blocks=4, hidden_d=16, n_heads=4, out_d=10).to(device)
    N_EPOCHS = 5

    optimizer = Adam(model.parameters(), lr=3e-4)
    criterion = CrossEntropyLoss()

    torch.cuda.empty_cache()

    for epoch in trange(N_EPOCHS, desc="Training"):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")



if __name__ == '__main__':
    main()




