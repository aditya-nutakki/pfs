import torch
import torch.nn as nn

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
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
    def __init__(self, input_channels = 3, output_channels = 3, num_steps = 512, down_factor = 1):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.down_factor = down_factor
    
        self.num_steps = num_steps
        self.embedding1 = nn.Embedding(self.num_steps, 512//self.down_factor)
        self.embedding2 = nn.Embedding(self.num_steps, 256//self.down_factor)

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
        _embedding1 = self.embedding1(t).view(-1, 512//self.down_factor, 1, 1)
        _embedding2 = self.embedding2(t).view(-1, 256//self.down_factor, 1, 1)
        # print(embedding.shape)
        s1, p1 = self.e1(inputs)
        # print(s1.shape, p1.shape)
        s2, p2 = self.e2(p1)
        # print(s2.shape, p2.shape)
        s3, p3 = self.e3(p2)
        p3 = p3 + _embedding2

        # print(s3.shape, p3.shape)
        s4, p4 = self.e4(p3)
        # print(s4.shape, p4.shape)
        p4 = p4 + _embedding1 

        b = self.b(p4)
        # print(b.shape)
        # print()
        # upsampling block
        d1 = self.d1(b, s4)
        # print(d1.shape)
        d1 = d1 + _embedding1
        # print(f"repeat {d1.shape}")
        d2 = self.d2(d1, s3)
        d2 = d2 + _embedding2
        # print(d2.shape)
        d3 = self.d3(d2, s2)
        # print(d3.shape)
        d4 = self.d4(d3, s1)
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
    y = model(inputs, randints)
    print(y.shape)
