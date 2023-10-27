import torch, torchvision
import os, json
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from helpers import ImageEncoder, TextEncoder

class CLIP(nn.Module):
    def __init__(self, embedding_dim = 512) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.image_encoder = ImageEncoder(self.embedding_dim)
        self.text_encoder = TextEncoder(self.embedding_dim)
        
    

    def forward(self, images, captions):
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(captions)

        return image_embeddings, text_embeddings


if __name__ == "__main__":
    batch_size = 4
    image_dims = (3, 224, 224)
    _dims = (batch_size, *image_dims)
    images = torch.randn(_dims)
    captions = torch.randint(0, 16, _dims)

    clip = CLIP()
    y = clip(images, captions)
    print(y, y.shape)