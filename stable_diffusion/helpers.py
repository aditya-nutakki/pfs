import torch, torchvision
import torch.nn.functional as F
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os, cv2
from random import choice
from torch.distributions.normal import Normal


class TextEncoder(nn.Module):
    def __init__(self, d_model = 128) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.embedding_proj = nn.Linear(768, self.d_model)
        self.dropout = nn.Dropout(0.2)
        self.act = nn.GELU()


    def forward(self, x):
        # captions to be in tensor form
        # print(x)
        x = self.model(input_ids = x["input_ids"], attention_mask = x["attention_mask"])
        # print(x.shape)
        x = x[0][:, 0] # distilbert output, can be written as x = x[0][:, 0, :] as well
        # print(x.shape)
        x = self.dropout(x)
        x = self.embedding_proj(x)
        return self.act(x)


class conv_block(nn.Module):
    def __init__(self, in_c, out_c, activation = "relu"):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()
        self.act = nn.ReLU() if activation == "relu" else nn.SiLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        # x = self.relu(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.relu(x)
        x = self.act(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, activation = "relu"):
        super().__init__()

        self.conv = conv_block(in_c, out_c, activation = activation)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, activation = "relu"):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c, activation = activation)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):
    def __init__(self, input_channels = 3, output_channels = 3, num_steps = 256, down_factor = 2, embedding_dim = 512):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.down_factor = down_factor
        self.embedding_dim = embedding_dim//self.down_factor
    
        self.num_steps = num_steps
        self.embedding = nn.Embedding(self.num_steps, self.embedding_dim)

        self.e1 = encoder_block(self.input_channels, 64//self.down_factor)
        self.e2 = encoder_block(64//self.down_factor, 128//self.down_factor)
        self.e3 = encoder_block(128//self.down_factor, 256//self.down_factor)
        self.e4 = encoder_block(256//self.down_factor, 512//self.down_factor)

        self.b = conv_block(512//self.down_factor, 1024//self.down_factor) # bottleneck

        self.d1 = decoder_block(1024//self.down_factor, 512//self.down_factor)
        self.d2 = decoder_block(512//self.down_factor, 256//self.down_factor)
        self.d3 = decoder_block(256//self.down_factor, 128//self.down_factor)
        self.d4 = decoder_block(128//self.down_factor, 64//self.down_factor)
    
        self.outputs = nn.Conv2d(64//self.down_factor, self.output_channels, kernel_size=1, padding=0)


    def forward(self, inputs, t = None):
        # downsampling block
        embedding = self.embedding(t).view(-1, self.embedding_dim, 1, 1)
        # print(embedding.shape)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        p4 = p4 + embedding 

        b = self.b(p4)
        
        # upsampling block
        d1 = self.d1(b, s4)
        d1 = d1 + embedding
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        return outputs




class VAE_Encoder(nn.Module):
    def __init__(self, input_shape = (3, 224, 224), output_shape = (3, 16, 16), latent_dims = 16) -> None:
        super().__init__()
        self.c, self.h, self.w = input_shape
        self.latent_dims = latent_dims
        self.k = 24

        self.model = nn.Sequential(
            nn.Conv2d(in_channels = self.c, out_channels = 32, kernel_size = 5),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 5),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.mean_layer, self.log_var_layer = nn.Linear(3*self.k*self.k, self.latent_dims), nn.Linear(3*self.k*self.k, self.latent_dims)

    @staticmethod
    def vae_sample(mean, log_var):
        bs, dim = mean.shape        
        epsilon = Normal(0, 1).sample((bs, dim))
        return mean + torch.exp(0.5 * log_var) * epsilon


    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        mean, log_var = self.mean_layer(x), self.log_var_layer(x)
        
        z = self.vae_sample(mean, log_var)
        print(f"latent space dims => {z.shape}")
        return z

    

class VAE_Decoder(nn.Module):
    def __init__(self, input_shape = (3, 24, 24), output_shape = (3, 224, 224), latent_dims = 16) -> None:
        super().__init__()
        self.k = 24
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 7, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 7, stride = 2, padding = 2, output_padding=1),
        )
        self.input_shape = input_shape
        self.latent_linear = nn.Linear(latent_dims, 3 * self.k * self.k)
        

    def forward(self, x):
        x = self.latent_linear(x)
        x = x.view(-1, *self.input_shape)
        x = self.model(x)
        print(x.shape)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dims = 16) -> None:
        super().__init__()
        self.latent_dims = latent_dims
        self.encoder = VAE_Encoder(latent_dims=self.latent_dims)
        self.decoder = VAE_Decoder(latent_dims=self.latent_dims)
        

    def forward(self, x):
        return self.decoder(self.encoder(x))





class CLIPDataset(Dataset):
    def __init__(self, dataset_path, _transforms = None, max_len = 32) -> None:
        super().__init__()
        
        self.max_len = max_len
        self.transforms = _transforms
        
        if not self.transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5, ), (0.5,))
            ])

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # print(type(self.tokenizer))

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
            # caption = captions[0]
            # print(f"caption is: {caption}")
            caption = self.tokenizer(caption, return_tensors = "pt", max_length = self.max_len, padding = "max_length")
            caption["input_ids"] = caption["input_ids"].view(-1)
            # print(caption)
        # print(self.tokenizer.batch_decode(caption["input_ids"], skip_special_tokens = True)) # -> to decode a tensor to list of decoded sentences

        return image, caption
    







if __name__ == "__main__":
    # ds_path = "/mnt/d/work/datasets/coco_captions"
    # ds = CLIPDataset(ds_path)

    # unet = UNet(down_factor=4)
    # print(sum([p.numel() for p in unet.parameters()]))
    # x, t = torch.randn(4, 3, 224, 224), torch.Tensor([11]).type(torch.LongTensor)
    # y = unet(x, t)
    # print(y.shape)

    # enc = VAE_Encoder()
    # print(sum([p.numel() for p in enc.parameters()]))
    # x = torch.randn(4, 3, 224, 224)
    # y = enc(x)
    # print(y.shape, y.mean(), y.std())

    # enc = VAE_Decoder()
    # print(sum([p.numel() for p in enc.parameters()]))
    # x = torch.randn(4, 16)
    # y = enc(x)
    # print(y.shape, y.mean(), y.std())

    vae = VAE()
    print(sum([p.numel() for p in vae.parameters()]))
    x = torch.randn(4, 3, 224, 224)
    y = vae(x)
    print(y.shape)
    pass