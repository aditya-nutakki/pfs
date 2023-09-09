import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sin, cos


d_model = 512
h = 8

assert d_model % h == 0, f"Cant divide {d_model} embedding dim by {h} heads"
dk, dv = d_model//h, d_model//h

n_layers = 2
dff = 2048
max_seq_len = 128
vocab_size = 128
target_vocab_size = 148


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size = vocab_size, d_model = d_model) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model, self.max_seq_len = d_model, max_seq_len
        self.pos_embedding = self.get_pos_embedding(self.max_seq_len, self.d_model)    

    def get_pos_embedding(self, max_seq_len = max_seq_len, d_model = d_model):
        # x is of shape (-1, seq_len, d_model); embeddings need to be performed at (-1, i, d_i)
        x = torch.zeros(max_seq_len, d_model)
        for i in range(len(x)):                         
            for j in range(0, len(x[0, :]), 2):
                x[i, j] = sin(i/10000**(2*j/14))    
                x[i, j + 1] = cos(i/10000**(2*j/14))
                # print(i, j, x[i,j], x[i, j+1])      
        
        x.requires_grad = False
        return x
        
    def forward(self, x):
        return self.pos_embedding + self.embedding(x) # * (d_model**0.5)



class FeedForward(nn.Module):
    def __init__(self, d_model = 512, dff = 2048) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, mask = None) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.25)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.mask = mask

        self.wo = nn.Linear(d_model, d_model)

        self.ff = FeedForward()


    def attention(self, q, k, v, mask=None, dropout=None):
        dot_product = (q @ k.transpose(-1, -2))/(dk**0.5)
        
        if mask is not None:
            # dot_product = dot_product.masked_fill(mask==0, -numpy.inf) # or you could use a very high negative value like -1e12 or something
            dot_product = torch.tril(dot_product)
            dot_product = dot_product.masked_fill(dot_product==0, float("-inf")) # or you could use a very high negative value like -1e12 or something
            # print(dot_product)
        dot_product = F.softmax(dot_product, dim = -1)

        if dropout:
            dot_product = self.dropout(dot_product)

        return dot_product @ v, dot_product
    

    def forward(self, q, k, v, mask = None, dropout = None):
        q = self.wq(q).view(-1, max_seq_len, h, dk).transpose(1, 2)
        k = self.wq(k).view(-1, max_seq_len, h, dk).transpose(1, 2) 
        v = self.wq(v).view(-1, max_seq_len, h, dv).transpose(1, 2)

        # q, k, v would now have the shape (batch_size, h, max_seq_len, dk)

        x, attention_score = self.attention(q,k,v, mask=self.mask)
        # print(x.shape, attention_score.shape)

        x = x.view(-1, max_seq_len, h * dk)
        x = self.wo(x)
        # is of the initial shape (-1, max_seq_len, d_model); which is the same initial shape. this can later be used to pass this to 'n' encode blocks
        return x


class EncoderBlock(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.mha = MultiHeadAttention()
        self.ff = FeedForward()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.layer_norm(x + self.mha(x, x, x))
        # print(f"Encoder shape => {x.shape}")
        x = self.layer_norm(x + self.ff(x))
        
        return x


class DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.masked_mha = MultiHeadAttention(mask=True)
        self.mha = MultiHeadAttention()

        self.layer_norm = nn.LayerNorm(d_model)
        self.ff = FeedForward()

    
    def forward(self, x, encoder_output):
        # x = self.emb(x)
        x = self.layer_norm(x + self.mha(x, x, x))
        x = self.layer_norm(x + self.masked_mha(x, encoder_output, encoder_output))
        x = self.layer_norm(x + self.ff(x))
        
        return x
