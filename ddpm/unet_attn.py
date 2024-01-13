import torch
import torch.nn as nn
import torch.nn.functional as F

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps = 1000, activation = "relu", embedding_dims = None):
        super().__init__()


        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.embedding_dims = embedding_dims if embedding_dims else out_c
        self.embedding = nn.Embedding(time_steps, embedding_dim = self.embedding_dims) # temporarily say number of positions is 512 by default, change it later. Ideally it should be num_time_steps from the ddpm
        self.relu = nn.ReLU()
        self.act = nn.ReLU() if activation == "relu" else nn.SiLU()

        
    def forward(self, inputs, time = None):

        time_embedding = self.embedding(time).view(-1, self.embedding_dims, 1, 1)
        # print(f"time embed shape => {time_embedding.shape}")
        x = self.conv1(inputs)
        x = self.bn1(x)
        # x = self.relu(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.relu(x)
        x = self.act(x)

        x = x + time_embedding
        # print(f"conv block {x.shape}")
        # print()
        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps, activation = "relu"):
        super().__init__()

        self.conv = conv_block(in_c, out_c, time_steps = time_steps, activation = activation, embedding_dims = out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs, time = None):
        x = self.conv(inputs, time)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps, activation = "relu"):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c, time_steps = time_steps, activation = activation, embedding_dims = out_c)

    def forward(self, inputs, skip, time = None):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x, time)

        return x



class AttnBlock(nn.Module):
    def __init__(self, embedding_dims, num_heads = 4) -> None:
        super().__init__()
        
        self.embedding_dims = embedding_dims
        self.ln = nn.LayerNorm(embedding_dims)

        self.mhsa = MultiHeadSelfAttention(embedding_dims = embedding_dims, num_heads = num_heads)

        self.ff = nn.Sequential(
            nn.LayerNorm(self.embedding_dims),
            nn.Linear(self.embedding_dims, self.embedding_dims),
            nn.GELU(),
            nn.Linear(self.embedding_dims, self.embedding_dims),
        )

    def forward(self, x):
        bs, c, sz, _ = x.shape
        x = x.view(-1, self.embedding_dims, sz * sz).swapaxes(1, 2) # is of the shape (bs, sz**2, self.embedding_dims)
        x_ln = self.ln(x)
        _, attention_value = self.mhsa(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, c, sz, sz)




class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dims, num_heads = 4) -> None:
        super().__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads

        assert self.embedding_dims % self.num_heads == 0, f"{self.embedding_dims} not divisible by {self.num_heads}"
        self.head_dim = self.embedding_dims // self.num_heads

        self.wq = nn.Linear(self.head_dim, self.head_dim)
        self.wk = nn.Linear(self.head_dim, self.head_dim)
        self.wv = nn.Linear(self.head_dim, self.head_dim)

        self.wo = nn.Linear(self.embedding_dims, self.embedding_dims)


    def attention(self, q, k, v):
        # no need for a mask
        attn_weights = F.softmax((q @ k.transpose(-1, -2))/self.head_dim**0.5, dim = -1)
        return attn_weights, attn_weights @ v        


    def forward(self, q, k, v):
        bs, img_sz, c = q.shape

        q = q.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v of the shape (bs, self.num_heads, img_sz**2, self.head_dim)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        attn_weights, o = self.attention(q, k, v) # of shape (bs, num_heads, img_sz**2, c)
        
        o = o.transpose(1, 2).contiguous().view(bs, img_sz, self.embedding_dims)
        o = self.wo(o)

        return attn_weights, o



class UNet(nn.Module):
    def __init__(self, input_channels = 3, output_channels = 3, time_steps = 512, down_factor = 1):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.down_factor = down_factor
        self.time_steps = time_steps
        self.time_steps = time_steps
        
        self.e1 = encoder_block(self.input_channels, 64, time_steps=self.time_steps)
        self.e2 = encoder_block(64, 128, time_steps=self.time_steps)
        # self.da2 = AttnBlock(128)

        self.e3 = encoder_block(128, 256, time_steps=self.time_steps)
        self.da3 = AttnBlock(256)

        self.e4 = encoder_block(256, 512, time_steps=self.time_steps)
        self.da4 = AttnBlock(512)
        
        self.b = conv_block(512, 1024, time_steps=self.time_steps) # bottleneck
        self.ba1 = AttnBlock(1024)

        self.d1 = decoder_block(1024, 512, time_steps=self.time_steps)
        self.ua1 = AttnBlock(512)

        self.d2 = decoder_block(512, 256, time_steps=self.time_steps)
        self.ua2 = AttnBlock(256)

        self.d3 = decoder_block(256, 128, time_steps=self.time_steps)
        # self.ua3 = AttnBlock(128)

        self.d4 = decoder_block(128, 64, time_steps=self.time_steps)
        # self.ua4 = AttnBlock(64)

        self.outputs = nn.Conv2d(64, self.output_channels, kernel_size=1, padding=0)


    def forward(self, inputs, t = None):
        # downsampling block
        # print(embedding.shape)
        s1, p1 = self.e1(inputs, t)
        # p1 = self.da1(p1)
        # print(s1.shape, p1.shape)
        s2, p2 = self.e2(p1, t)
        # p2 = self.da2(p2)
        # print(s2.shape, p2.shape)
        s3, p3 = self.e3(p2, t)
        p3 = self.da3(p3)
        # print(s3.shape, p3.shape)
        s4, p4 = self.e4(p3, t)
        p4 = self.da4(p4)
        # print(s4.shape, p4.shape)

        b = self.b(p4, t)
        b = self.ba1(b)
        # print(b.shape)
        # print()
        # upsampling block
        d1 = self.d1(b, s4, t)
        d1 = self.ua1(d1)
        # print(d1.shape)
        # print(f"repeat {d1.shape}")
        d2 = self.d2(d1, s3, t)
        d2 = self.ua2(d2)
        # print(d2.shape)
        d3 = self.d3(d2, s2, t)
        # d3 = self.ua3(d3)
        # print(d3.shape)
        d4 = self.d4(d3, s1, t)
        # d4 = self.ua4(d4)
        # print(d4.shape)

        outputs = self.outputs(d4)
        # print(outputs.shape)
        return outputs


if __name__ == "__main__":    
    device = "cuda:0"
    batch_size = 2
    in_channels, w = 3, 128
    inputs = torch.randn((batch_size, in_channels, w, w), device=device)
    randints = torch.randint(1, 512, (batch_size, ), device=device)
    model = UNet().to(device)
    print(f"model has {sum([p.numel() for p in model.parameters()])} params")
    y = model(inputs, randints)
    print(y.shape)
    
    # x = torch.randn(4, 32, 16, 16)
    # mhsa = AttnBlock(32, 4)
    # y = mhsa(x)
    # print(y.shape, x.shape)

