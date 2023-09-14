import torch, torchvision
import os
import torch.nn as nn
import torch.nn.functional as F
from math import sin, cos

d = 32 # 768 is the number used in the paper
dff = 3072
h = 4



class MultiHeadAttention(nn.Module):
    def __init__(self, h = 8, d = 32, patch_dim = 7) -> None:
        super().__init__()
        self.d = d
        assert d % h == 0, f"cant divide {d} hidden dimensions by {h} heads"
        self.dk = self.d // h
        self.dv = self.d // h
        self.dropout = nn.Dropout(0.25)
        self.n_patch = patch_dim**2 + 1

        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        
        self.wo = nn.Linear(d, d)


    def attention(self, q, k, v):
        dot_product = F.softmax((q @ k.transpose(-1, -2))/self.dk**0.5, dim = -1)
        return dot_product @ v, dot_product


    def forward(self, q, k, v):
        # q, k, v to be of the shape => (-1, 49 + 1, d_hidden) -> (-1, h, 49+1, dk)
        q = self.wq(q).view(-1, self.n_patch, h, self.dk).transpose(1, 2)
        k = self.wk(k).view(-1, self.n_patch, h, self.dk).transpose(1, 2)
        v = self.wv(v).view(-1, self.n_patch, h, self.dv).transpose(1, 2)
        
        attention, dot_product = self.attention(q, k, v)
        # attention to be of the shape (-1, h, self.n_patch, dk)
        attention = attention.transpose(1, 2).reshape(-1, self.n_patch, d) # d can also be written as h * self.dv
        return self.wo(attention), dot_product
    


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dff = dff
        self.d = d
        self.linear1 = nn.Linear(d, dff)
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(dff, d)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


class EncoderBlock(nn.Module):
    def __init__(self, h, d, patch_dim) -> None:
        super().__init__()
        self.dff = dff
        self.h = h
        self.d = d
        self.patch_dim = patch_dim
        self.layer_norm = nn.LayerNorm(self.d)
        self.mha = MultiHeadAttention(h = self.h, d = self.d, patch_dim = self.patch_dim)
        self.ff = FeedForward()

    def forward(self, x):
        normal_x = self.layer_norm(x)
        _attention_output, dot_product = self.mha(normal_x, normal_x, normal_x)
        x = x + _attention_output
        x = x + self.ff(self.layer_norm(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, n_layers, patch_dim) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.patch_dim = patch_dim
        self.encoders = nn.ModuleList([EncoderBlock(h = 4, d = 32, patch_dim=self.patch_dim) for _ in range(n_layers)])

    def forward(self, x):
        for e in self.encoders:
            x = e(x)
        return x



class ViT(nn.Module):
    def __init__(self, image_dims = (1, 28, 28), num_patches = 4, num_classes = 10) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.image_dims = image_dims 
        self.c, self.h, self.w = image_dims
        self.d = d
        self.num_classes = num_classes
        assert self.image_dims[1] == self.image_dims[2]
        assert self.image_dims[1] % num_patches == 0, f"cannot divide {self.image_dims[1]} with patch size of {num_patches}"
        
        self.N = (self.h * self.w)//self.num_patches**2
        self.n_patches = self.h // self.num_patches
        self.patch_res = self.num_patches**2 * self.c # num of pixels in each patch (patch and image must be a square) -> this case its a 7x7 patch dim
        
        print(self.N, self.patch_res)

        self.linear_mapping = nn.Linear(self.patch_res, self.d)
        self.class_token = nn.Parameter(torch.randn(1, self.d))

        self.pos_embedding = self.get_pos_embedding(self.n_patches ** 2 + 1, self.d) # add this to the tensor of (self.N^2 + 1 class_token) 
        # print(type(self.pos_embedding)) # type is of torch.nn.parameter.Parameter
        self.layer_norm = nn.LayerNorm(self.d)
        self.vit_encoder = ViTEncoder(n_layers=3, patch_dim=self.n_patches)

        self.mlp_ff = nn.Linear(self.d, self.d)
        self.class_ff = nn.Linear(self.d, self.num_classes)

    def get_pos_embedding(self, n, d):
        pos_embedding = torch.zeros(n, d)

        for i in range(n):
            for j in range(d):
                pos_embedding[i][j] = sin(i/ 10000**(j/d)) if j % 2 == 0 else cos(i/ 10000**(j-1/d))
                
        pos_embedding = nn.Parameter(pos_embedding)
        pos_embedding.requires_grad = False

        return pos_embedding


    def forward(self, x):
        x = x.view(-1, self.N, self.patch_res)
        # x = F.relu(self.linear_mapping(x))
        x = self.linear_mapping(x)
        x = torch.stack(
            [torch.cat([self.class_token, _x], dim = 0) for _x in x] # we add the class_token embedding for each image coming in; we need to iterate over batch_size so thaat all of them get it
        )
        x += self.pos_embedding
        # you now have a tensor of shape (batch_size, n_patches**2 + 1, d); we must now normalise and then pass it through the MHA module -> this can be encapsulated in a single encoder block
        x = self.vit_encoder(x)

        x = self.mlp_ff(x[:, -1])
        return F.softmax(self.class_ff(x), dim = -1)




if __name__ == "__main__":
    w = 28
    image_dims = (1, w, w) # c, h, w
    vit = ViT(image_dims= image_dims, num_patches = 4, num_classes=10)
    img_dims = (4, *image_dims) # n, c, h, w
    img = torch.rand(img_dims)
    y = vit(img)
    print(y.shape)
