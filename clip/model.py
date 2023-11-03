import torch, torchvision
import os, json
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from helpers import ImageEncoder, TextEncoder


class CLIP(nn.Module):
    def __init__(self, embedding_dim = 512, model_type = "resnet") -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        self.image_encoder = ImageEncoder(embedding_dim = self.embedding_dim, model_type = self.model_type)
        self.text_encoder = TextEncoder(d_model = self.embedding_dim)
        
        assert self.image_encoder.embedding_dim == self.text_encoder.d_model, f"embedding_dim of {self.embedding_dim} not constant across image and text encoders"
    

    def forward(self, images, captions):
        image_embeddings = self.image_encoder(images)
        # print(f"image embedding shape => {image_embeddings.shape}")
        text_embeddings = self.text_encoder(captions)
        # print(f"text embedding shape => {text_embeddings.shape}")
        
        # normalizing tensors
        # l2 normalize a vector by b = a / torch.sqrt(torch.sum(a**2, dim=-1)).view(-1, 1)
        image_embeddings, text_embeddings = image_embeddings/ torch.sqrt(torch.sum(image_embeddings**2, dim=-1)).view(-1, 1), text_embeddings / torch.sqrt(torch.sum(text_embeddings**2, dim=-1)).view(-1, 1)
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
    