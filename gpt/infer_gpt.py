import torch.nn as nn
import torch.nn.functional as F
import os, json
import torch
from gpt import GPT, Utils, get_pos_embedding
from time import time

loader = Utils()
model_path = "./tinys_rand_22500_128_256_8_1024_8.pt"

device = "cuda"


def encode(string):
    # converts string to ints
    return [stoi[s] for s in string]


def decode(ints, return_raw = False):
    # converts ints to string
    _decoded = [itos[i] for i in ints]
    return _decoded if return_raw else "".join(_decoded)



# def generate(model, start_token = " ", num_tokens = 256, temperature = 1.0):
#     model.eval()
#     model.to(device)

#     print(model.max_len)
#     with torch.no_grad():
#         pos_embeddings = get_pos_embedding(max_len = 128, d_model = model.d_model, device = device)
#         tokens = encode(start_token)
#         all_tokens = tokens
#         print(tokens)
#         for i in range(num_tokens):
#             # fix case of handling more than max_len tokens; ignoring it for now
            
#             # x = loader.tensor_encode(op).to(device)

#             x = torch.tensor(tokens, dtype = torch.long).to(device)
#             x = model.embedding(x)
#             _pos_embeddings = pos_embeddings[:len(tokens)] if i < model.max_len else pos_embeddings

#             x = x + _pos_embeddings
#             # print(x.shape) # is of the shape (num_chars, d_model)
#             x = torch.unsqueeze(x, dim = 0)
#             for block in model.decoders:
#                 x = block(x)
            
#             x = model.lm_head(x)
#             x = x[:, -1, :] / temperature

#             probs = F.softmax(x, dim = -1)

#             # sampling 
#             chosen_token = torch.multinomial(probs, num_samples=1)
#             # print(chosen_token, type(chosen_token))
#             # op = op + loader.tensor_decode(chosen_token)
#             # x = x + 
#             tokens = tokens + chosen_token[0].tolist()
#             tokens = tokens[-128:]
#             all_tokens = all_tokens + chosen_token[0].tolist()
#         print("done")
#         print()
#         print(decode(all_tokens))
#         print()
#         print(decode(tokens))


def generate_one_token(context, model):
    pass



def generate(model, start_token = " ", num_tokens = 256, temperature = 1.0):
    model.eval()
    model.to(device)
    all_tokens = encode(start_token)
    window = all_tokens
    pred_tokens = []
    position_embeddings = get_pos_embedding(max_len = model.max_len, d_model = model.d_model, device = device)

    for i in range(num_tokens):
        with torch.no_grad():

            # if len(window) >= model.max_len:
            #     pass
            window = all_tokens[-model.max_len : ]

            x = torch.tensor(window, dtype = torch.long, device = device)
            x = model.embedding(x)
            x = x + position_embeddings[:len(window)]
            x = torch.unsqueeze(x, dim = 0)

            for block in model.decoders:
                x = block(x)
            
            x = model.lm_head(x)
            x = x[:, -1, :]/temperature

            probs = F.softmax(x, dim = -1)

            chosen_token = torch.multinomial(probs, num_samples = 1)[0].tolist() # will be an idx
            pred_tokens += chosen_token
            all_tokens += chosen_token

    return decode(all_tokens)
    # print()
    # print(decode(pred_tokens))

# def generate(model, start_token = " ", num_tokens = 111, temperature = 1.0):
#     model.eval()
#     model.to(device)
#     with torch.no_grad():
#         token = torch.tensor(encode(start_token), dtype=torch.long).view(1,-1).to(device)
#         full_token = torch.tensor(encode(start_token), dtype=torch.long).view(1,-1).to(device)
#         position_embeddings = get_pos_embedding(max_len = model.max_len, d_model = model.d_model, device = device)
        
#         # print(token.shape)
#         # print(token)
#         for l in range(len(start_token), num_tokens - 1):
#             # print(token.shape, l)
#             if l > model.max_len:
#                 x = x[:, abs(l-model.max_len):]
            
#             # ops, _ = model(token)
#             x = model.embedding(token)
#             x = x + position_embeddings[:len(x)]
#             # token = torch.unsqueeze(x, dim = 0)

#             for block in model.decoders:
#                 x = block(x)
            
#             x = model.lm_head(x)
#             # print(token, token.shape)
#             x = x[:, [-1], :]/temperature
            
#             ops = x[:, -1, :]
#             ops = F.softmax(ops, dim = -1)
#             # print(f"ops shape {ops.shape}")
#             most_likely_char = torch.multinomial(ops, num_samples = 1)
#             print(token.shape, most_likely_char.shape)
#             token = torch.cat((token, most_likely_char), dim = 1)
#             full_token = torch.cat((token, most_likely_char), dim = 1)
        
#         # print(token)
#     return decode(full_token.cpu().detach().numpy().flatten().tolist())


if __name__ == "__main__":
    ckpt = torch.load(model_path)
    stoi = ckpt["vocab"]

    model = GPT(vocab = stoi, max_len = 128, num_layers = 8, d_model = 256, n_heads = 8, dff = 1024)
    model.load_state_dict(ckpt["state_dict"])
    
    itos = {i : _char for _char, i in stoi.items()}
    print(stoi)
    # print(itos)
    print()
    
    stime = time()
    generated = generate(model = model, start_token = "What would", num_tokens= 512, temperature= 0.5)
    ftime = time()

    print(generated)
    print(f"\ngenerated sample in {ftime - stime}s")

