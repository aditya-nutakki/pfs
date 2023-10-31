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
        self.image_encoder = ImageEncoder(embedding_dim = self.embedding_dim)
        self.text_encoder = TextEncoder(d_model = self.embedding_dim)
        
        assert self.image_encoder.embedding_dim == self.text_encoder.d_model, f"embedding_dim of {self.embedding_dim} not constant across image and text encoder"
    

    def forward(self, images, captions):
        image_embeddings = self.image_encoder(images)
        # print(f"image embedding shape => {image_embeddings.shape}")
        text_embeddings = self.text_encoder(captions)
        # print(f"text embedding shape => {text_embeddings.shape}")

        return image_embeddings, text_embeddings


if __name__ == "__main__":
    batch_size, max_len = 4, 16
    image_dims = (3, 224, 224)
    _dims = (batch_size, *image_dims)
    images = torch.randn(_dims)
    captions = torch.randint(0, 16, (batch_size, max_len))

    clip = CLIP(embedding_dim = 128)
    iemb, temb = clip(images, captions)
    print(iemb.shape, temb.shape)
    