from cgan import Generator, Z_DIM, num_classes, BATCH_SIZE, EPOCH_NUM, min_num, max_num
import torchshow as ts
import torch

device = "cuda"
# model_path = "./generator.pt"
model_path = "./generator_new.pt"

noise = torch.randn(BATCH_SIZE//4, Z_DIM, 1, 1, device = device)
ckpt = torch.load(model_path)
# print(ckpt, type(ckpt))

model = Generator().to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with torch.no_grad():
    for i in range(10):
        classes = (torch.ones(BATCH_SIZE//4, device=device) * i).long().to(device)
        preds = model(noise, classes)
        ts.save(preds, f"infer_{min_num}_{max_num}_{EPOCH_NUM}_{i}.jpeg")