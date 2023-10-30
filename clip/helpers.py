from torch.utils.data import Dataset, DataLoader
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import os
from collections import OrderedDict

from torchvision.models.vision_transformer import VisionTransformer

from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer


class ImageEncoder(nn.Module):
    def __init__(self, image_size = 224, patch_size = 16, num_layers = 6, num_heads = 8,
                 hidden_dim = 256, mlp_dim = 1024, attention_dropout = 0.15, dropout = 0.15, num_classes = 0, embedding_dim = 256) -> None:
        super().__init__()
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes

        self.model = VisionTransformer(self.image_size, self.patch_size, self.num_layers, self.num_heads, self.hidden_dim, self.mlp_dim, 
                                       self.dropout, self.attention_dropout, self.num_classes)
        self._modify_vit()

    def _modify_vit(self):
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["pre_logits"] = nn.Linear(self.model.hidden_dim, self.embedding_dim)
        heads_layers["act"] = nn.Tanh() # do try with other activation functions
        
        self.model.heads = nn.Sequential(heads_layers)

    def forward(self, images):
        return self.model(images)
    


class TextEncoder(nn.Module):
    def __init__(self, vocab_size , max_seq_len = 8, num_layers = 4, nhead = 8, d_model = 128) -> None:
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = self.nhead)
        # self.model = nn.TransformerEncoder(encoder_layer = self.encoder_layer, num_layers = self.num_layers)
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.embedding_proj = nn.Linear(768, self.d_model)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        # captions to be in tensor form
        x = self.model(x)
        # print(x.shape)
        x = x[0][:, 0] # distilbert output
        # print(x.shape)
        x = self.dropout(x)
        x = self.embedding_proj(x)
        return x



class CLIPDataset(Dataset):
    def __init__(self, dataset_path) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.classes = os.listdir(self.dataset_path)
        self.num_classes = len(self.classes)


    def __len__(self):
        return self.num_classes
    
    def __getitem__(self, index):
        # to return a pair of (images, text-caption)
        return 0


if __name__ == "__main__":
    # enc = ImageEncoder()
    enc = TextEncoder(vocab_size = 10)
    print(sum([p.numel() for p in enc.parameters()]))
    
    # x = torch.randn(4, 3, 224, 224)
    # y = enc(x)
    # print(y, y.shape)
    # print(torch.min(y), torch.max(y))

    x = torch.randint(0, 10, (4, 12))
    print(x)
    y = enc(x)
    # hidden_state = y.last_hidden_state
    # print(hidden_state.shape)
    print(y, y.shape)
    # print(y)