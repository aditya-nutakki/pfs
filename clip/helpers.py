from torch.utils.data import Dataset, DataLoader
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import os, cv2
from collections import OrderedDict
from random import choice

from torchvision.models.vision_transformer import VisionTransformer
from torchvision.transforms import transforms

from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer
import torchshow as ts


class ImageEncoder(nn.Module):
    def __init__(self, image_size = 224, patch_size = 16, num_layers = 6, num_heads = 8,
                 hidden_dim = 256, mlp_dim = 1024, attention_dropout = 0.15, dropout = 0.15, num_classes = 0, embedding_dim = 128) -> None:
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
    def __init__(self, max_seq_len = 8, num_layers = 4, nhead = 8, d_model = 128) -> None:
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.d_model = d_model
        self.num_layers = num_layers

        self.encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = self.nhead)
        # self.model = nn.TransformerEncoder(encoder_layer = self.encoder_layer, num_layers = self.num_layers)
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.embedding_proj = nn.Linear(768, self.d_model)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        # captions to be in tensor form
        x = self.model(x)
        # print(x.shape)
        x = x[0][:, 0] # distilbert output, can be written as x = x[0][:, 0, :] as well
        # print(x.shape)
        x = self.dropout(x)
        x = self.embedding_proj(x)
        return x



class CLIPDataset(Dataset):
    def __init__(self, dataset_path, _transforms = None) -> None:
        super().__init__()
        
        self.transforms = _transforms
        
        if not self.transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5, ), (0.5,))
            ])

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print(type(self.tokenizer))

        self.dataset_path = dataset_path
        self.images_path, self.captions_path = os.path.join(self.dataset_path, "images"), os.path.join(self.dataset_path, "captions")
        self.images, self.captions = os.listdir(self.images_path), os.listdir(self.captions_path)

        self.images = [os.path.join(self.images_path, image) for image in self.images]
        self.captions = [os.path.join(self.captions_path, caption) for caption in self.captions]


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # to return a pair of (images, text-caption)
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image.shape)
        image = self.transforms(image)

        # tokenize text
        caption_path = self.captions[index]
        # print(caption_path)
        with open(caption_path) as f:
            captions = f.readlines()
            captions = [caption.strip() for caption in captions]
            caption = choice(captions)
            print(f"caption is: {caption}")
            caption = self.tokenizer(caption, return_tensors = "pt", max_length = 128, padding = "max_length")
            caption = caption["input_ids"].view(-1)
            # print(caption)
        # print(self.tokenizer.batch_decode(caption["input_ids"], skip_special_tokens = True)) # -> to decode a tensor to list of decoded sentences

        return image, caption


if __name__ == "__main__":
    # enc = ImageEncoder()
    enc = TextEncoder()
    # print(sum([p.numel() for p in enc.parameters()]))
    
    # x = torch.randn(4, 3, 224, 224)
    # y = enc(x)
    # print(y, y.shape)
    # print(torch.min(y), torch.max(y))

    # x = torch.randint(0, 10, (4, 12))
    # print(x)
    # y = enc(x)
    # print(y, y.shape)
    # print(y)
    data = CLIPDataset("/mnt/d/work/datasets/coco_captions")
    image, caption = data[23]
    image1, caption2 = data[2]
    print(caption.shape)
    caption = torch.cat([caption, caption2], dim = 0)
    print(caption.shape)
    # print(image.shape, caption)
    # ts.save(image, "dummy.jpeg")
    y = enc(caption)
    print(y, y.shape)