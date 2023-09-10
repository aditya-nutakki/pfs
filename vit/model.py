import torch, torchvision
import os
import torch.nn as nn
import torch.nn.functional as F
from math import sin, cos

d = 32 # 768 is the number used in the paper
dff = 3072
h = 12


class ViT(nn.Module):
    def __init__(self, image_dims = (1, 28, 28), num_patches = 4) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.image_dims = image_dims 
        self.c, self.h, self.w = image_dims
        self.d = d

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


    def get_pos_embedding(n, d):
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
        
        return x




if __name__ == "__main__":
    w = 28
    image_dims = (1, w, w) # c, h, w
    vit = ViT(image_dims= image_dims, num_patches = 4)
    img_dims = (4, *image_dims) # n, c, h, w
    img = torch.rand(img_dims)
    y = vit(img)
    print(y.shape)
