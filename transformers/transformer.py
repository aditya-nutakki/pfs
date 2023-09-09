import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# from torch.optim import Adam
from math import sin, cos
from helpers import *

target_vocab_size = 196

class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_embedding, self.output_embedding = InputEmbedding(), InputEmbedding(vocab_size = target_vocab_size)
        self.encoders = nn.ModuleList([EncoderBlock() for _ in range(n_layers)])
        self.decoders = nn.ModuleList([DecoderBlock() for _ in range(n_layers)])
        self.final_linear = nn.Linear(d_model, vocab_size)

    def encode(self, x):
        for e in self.encoders:
            x = e(x)
        return x

    def decode(self, y, encoder_output):
        for d in self.decoders:
            y = d(y, encoder_output)
        return y
        
    def forward(self, x, target = None):
        
        x = self.input_embedding(x)
        encoder_output = self.encode(x)

        if target is not None: # only for training; will have to make changes for inference
            y = self.output_embedding(target)
        
        y = self.decode(y, encoder_output)
        y = self.final_linear(y)
        return F.log_softmax(y, dim = -1)



if __name__ == "__main__":
    x = torch.randint(low = 0, high = vocab_size, size =(4, max_seq_len))
    _x = torch.randint(low = 0, high = target_vocab_size, size =(4, max_seq_len)) # expected output tensor
    # dec = DecoderBlock()
    # y = dec(x)
    # print(y.shape)
    trans = Transformer()
    y = trans(x, _x)
    print(sum(p.numel() for p in trans.parameters()))
    print(y.shape)

