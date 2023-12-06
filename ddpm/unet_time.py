import torch
import torch.nn as nn

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, activation = "relu", embedding_dims = None):
        super().__init__()


        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.embedding_dims = embedding_dims if embedding_dims else out_c
        self.embedding = nn.Embedding(512, embedding_dim = self.embedding_dims) # temporarily say number of positions is 512 by default, change it later. Ideally it should be num_time_steps from the ddpm
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
    def __init__(self, in_c, out_c, activation = "relu"):
        super().__init__()

        self.conv = conv_block(in_c, out_c, activation = activation, embedding_dims = out_c)
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
    def __init__(self, in_c, out_c, activation = "relu"):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c, activation = activation, embedding_dims = out_c)

    def forward(self, inputs, skip, time = None):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x, time)

        return x


class UNet(nn.Module):
    def __init__(self, input_channels = 3, output_channels = 3, num_steps = 512, down_factor = 1):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.down_factor = down_factor
    
        self.num_steps = num_steps
        # self.embedding = nn.Embedding(self.num_steps, 512)

        self.e1 = encoder_block(self.input_channels, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        self.b = conv_block(512, 1024) # bottleneck

        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
    
        self.outputs = nn.Conv2d(64, self.output_channels, kernel_size=1, padding=0)


    def forward(self, inputs, t = None):
        # downsampling block
        # print(embedding.shape)
        s1, p1 = self.e1(inputs, t)
        # print(s1.shape, p1.shape)
        s2, p2 = self.e2(p1, t)
        # print(s2.shape, p2.shape)
        s3, p3 = self.e3(p2, t)
        # print(s3.shape, p3.shape)
        s4, p4 = self.e4(p3, t)
        # print(s4.shape, p4.shape)

        b = self.b(p4, t)
        # print(b.shape)
        # print()
        # upsampling block
        d1 = self.d1(b, s4, t)
        # print(d1.shape)
        # print(f"repeat {d1.shape}")
        d2 = self.d2(d1, s3, t)
        # print(d2.shape)
        d3 = self.d3(d2, s2, t)
        # print(d3.shape)
        d4 = self.d4(d3, s1, t)
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
