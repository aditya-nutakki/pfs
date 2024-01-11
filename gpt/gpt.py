import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sin, cos
import numpy as np

max_len = 64


class Utils:
    def __init__(self, path = "./tinys.txt") -> None:
        
        self.path = path
        self.data, self.stoi = self.get_dataset(self.path)
        self.itos = {i : _char for _char, i in self.stoi.items()}

        self.vocab_size = len(self.stoi)
        # print(self.stoi)
        # print(self.itos)

    def get_dataset(self, path = "./tinys.txt"):
        with open(path, "r") as f:
            # f = f.read().lower()
            f = f.read()

        vocab = {_char: i for i, _char in enumerate(list(set(f)))} # returns stoi mapping
        return f, vocab


    def encode(self, string):
        # converts string to ints
        return [self.stoi[s] for s in string]


    def decode(self, ints, return_raw = False):
        # converts ints to string
        _decoded = [self.itos[i] for i in ints]
        return _decoded if return_raw else "".join(_decoded)


    def tensor_encode(self, string):
        encoded_string = self.encode(string)
        return torch.tensor(encoded_string, dtype = torch.long)


    def tensor_decode(self, token):
        token = token.detach().cpu().numpy()
        return self.decode(token)


    def batch_decode(self, batch):
        decoded = []
        for _item in batch:
            decoded.append(self.decode(_item))

        return decoded

    # def get_batch(self, batch_size, max_len):
    #     # seeds = np.random.randint(0, len(self.data) - max_len, (batch_size, ))
    #     x, y = [], []

    #     for _ in range(batch_size):
    #         rand_pos = np.random.randint(0, len(self.data) - max_len)
    #         x.append(torch.tensor(self.encode(self.data[rand_pos: rand_pos + max_len]), dtype = torch.long))
    #         # y.append(torch.tensor(self.encode(self.data[rand_pos + 1 : rand_pos + max_len + 1]), dtype = torch.long))
    #         y.append(torch.tensor(self.encode(self.data[rand_pos + max_len + 1]), dtype = torch.long))
        
    #     # return torch.stack(x, dim = 0), torch.stack(y, dim = 0)
    #     return torch.stack(x, dim = 0), torch.tensor(y, dtype = torch.long)

    def get_batch(self, batch_size, max_len):
        # seeds = np.random.randint(0, len(self.data) - max_len, (batch_size, ))
        x, y = [], []
        rand_len = np.random.randint(1, max_len + 1)
        for _ in range(batch_size):
            rand_pos = np.random.randint(0, len(self.data) - rand_len - 1)
            x.append(self.encode(self.data[rand_pos: rand_pos + rand_len]))
            # y.append(torch.tensor(self.encode(self.data[rand_pos + 1 : rand_pos + max_len + 1]), dtype = torch.long))
            y.append(self.encode(self.data[rand_pos + rand_len])[0])
        
        return torch.tensor(x, dtype = torch.long), torch.tensor(y, dtype = torch.long)


class FeedForward(nn.Module):
    def __init__(self, d_model, dff) -> None:
        super().__init__()
        self.d_model, self.dff = d_model, dff
        
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.dff),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.dff, self.d_model)
        )

    def forward(self, x):
        return self.ff(x)



class DecoderBlock(nn.Module):
    def __init__(self, d_model, dff = 512, n_heads = 4) -> None:
        super().__init__()
        self.d_model, self.dff, self.n_heads = d_model, dff, n_heads
        self.mhsa = SelfAttention(self.d_model, n_heads = self.n_heads)
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        self.ff = FeedForward(d_model = self.d_model, dff = self.dff)
        self.layer_norm2 = nn.LayerNorm(self.d_model)


    def forward(self, x):
        # x is of the shape (bs, max_len, d_model)
        _, attn = self.mhsa(x, x, x)
        x = x + attn
        x = self.layer_norm(x)

        x = x + self.ff(x)
        x = self.layer_norm2(x)

        return x


def get_pos_embedding(max_len, d_model, device):
    pos_embedding = torch.zeros(max_len, d_model)

    for pos, word_vector in enumerate(pos_embedding):
        for i, _ in enumerate(word_vector):     
            pos_embedding[pos][i] = sin(pos/10000**(i//2/d_model)) if i % 2 == 0 else cos(pos/10000**(i//2/d_model)) # dividing it //2 because it's 2*i in the paper.
           
    return pos_embedding.to(device = device)


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        assert self.d_model % self.n_heads == 0, f"{self.d_model} dims need to be divisible by n_heads {self.n_heads}"
        self.head_dim = self.d_model // self.n_heads

        self.wq = nn.Linear(self.head_dim, self.head_dim)
        self.wv = nn.Linear(self.head_dim, self.head_dim)
        self.wk = nn.Linear(self.head_dim, self.head_dim)

        self.wo = nn.Linear(self.d_model, self.d_model)


    def attention(self, q, k, v):
        # masking is to be done by default since this is a decoder only model !
        
        attn_weights = (q @ k.transpose(-1, -2))/self.head_dim**0.5 # should be of the shape (bs, n_heads, max_len, head_dim) @ (bs, n_heads, max_len, head_dim).transpose(-1, -2) = (bs, n_heads, max_len, max_len)
        attn_weights = torch.tril(attn_weights)
        attn_weights = F.softmax(torch.masked_fill(attn_weights, attn_weights == 0, -torch.inf), dim = -1)
        # print("attn weights !")
        # print(attn_weights.std())
        return attn_weights, attn_weights @ v


    def forward(self, q, k, v):
        bs, max_len, d_model = q.shape # assume inputs q, k, v are of the same dims
        # q, k, v are of shapes (bs, max_len, d_model)

        # print(f"init shape for q, k, v: {q.shape}, {k.shape}, {v.shape}")

        q = q.view(bs, max_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, max_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, max_len, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v are of shapes (bs, self.n_heads, max_len, self.head_dim)
        # print(q.shape, k.shape, v.shape)

        q, k, v = self.wq(q), self.wk(k), self.wv(v)
        # print(q.shape, k.shape, v.shape)

        attn_weights, o = self.attention(q, k, v)
        # print(f"attn weights shape {attn_weights.shape}; o shape {o.shape}")
        o = o.transpose(1, 2).contiguous().view(bs, max_len, d_model)
        o = self.wo(o)
        # print(f"final o shape {o.shape}")
        return attn_weights, o


class GPT(nn.Module):
    def __init__(self, vocab, max_len = 128, num_layers = 4, d_model = 128, n_heads = 4, dff = 512) -> None:
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.num_layers = num_layers
        self.d_model, self.dff = d_model, dff
        self.max_len = max_len
        self.n_heads = n_heads

        self.embedding = nn.Embedding(self.vocab_size, self.d_model) # of the shape (65, 128)

        self.decoders = nn.ModuleList([DecoderBlock(d_model = self.d_model, dff = self.dff) for _ in range(num_layers)])
        self.pos_embedding = get_pos_embedding(max_len = self.max_len, d_model = self.d_model, device = "cuda")

        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x):
        x = self.embedding(x) 
        x = x + self.pos_embedding[:x.shape[1]] # of the shape (-1, max_len, d_model)

        for block in self.decoders:
            x = block(x)
        
        x = self.lm_head(x) # logits would be of shape (-1, max_len, vocab_size)
        return x[:, -1, :] # returning logits of shape (bs, vocab_size)


device = "cuda"

def train():
    loader = Utils()

    max_len = 128
    d_model, dff, n_heads, num_layers = 256, 1024, 8, 8
    # model = GPT(vocab = loader.stoi, max_len = max_len, d_model = 256, num_layers = 8, dff = 1024, n_heads = 8)
    # model.load_state_dict(torch.load("./tinys_rand_22500_128_256_8_1024_8.pt")["state_dict"])

    model_path = "./tinys_rand_22500_128_256_8_1024_8.pt"
    ckpt = torch.load(model_path)
    stoi = ckpt["vocab"]

    model = GPT(vocab = stoi, max_len = 128, num_layers = 8, d_model = 256, n_heads = 8, dff = 1024)
    model.load_state_dict(ckpt["state_dict"])


    model = model.to(device)

    print(f'instantiated model !')
    print(f"model has {sum([p.numel() for p in model.parameters()])} parameters")

    opt = torch.optim.Adam(model.parameters(), lr = 3e-4)

    i, steps = 0, 100000
    losses = []

    print("starting to train ...")
    while i < steps:
        x, y = loader.get_batch(batch_size = 32, max_len = max_len)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        # probs = F.softmax(logits, dim = -1)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        # print(loss.item())
        if i % 500 == 0:
            print(f"loss at step {i}; {loss}")
            ckpt = {
                "state_dict": model.state_dict(),
                "vocab": model.vocab
            }
            torch.save(ckpt, f"./tinys_rand_{i}_{max_len}_{d_model}_{num_layers}_{dff}_{n_heads}.pt")
            print(f"saved model !")

        i += 1




if __name__ == "__main__":
    # u = Utils()
    # sample = "hello world !"
    # print(u.decode(u.encode(sample)))
    
    # pos_embedding = get_pos_embedding(max_len = 100, d_model = 512)
    # print(pos_embedding)
    # pos_embedding = pos_embedding.detach().cpu().numpy()
    # cv2.imshow("embedding window", pos_embedding)
    # cv2.waitKey(0)

    # x = torch.randn(4, 8, 16)
    # sa = SelfAttention(16, 4)
    # print(sa(x, x, x))

    # x = torch.randn(4, 12, 32)
    # d = nn.ModuleList([DecoderBlock(d_model = 32, dff = 128) for _ in range(4)])
    # for i, b in enumerate(d):
    #     x = b(x)
    #     print(i)
    # print(x.shape, x.std())

    # gpt = GPT(
    #     vocab_size=65,
    #     max_len=32
    # )

    # x = torch.randint(0, 65, (4, 32))
    # print(x.shape)
    # y = gpt(x)
    # print(y.shape, y.std())

    # loader = Utils()
    # x, y = loader.get_batch(batch_size = 4, max_len = 64)
    # print(x, x.shape)
    # print()
    # print(y, y.shape)
    
    train()
    
