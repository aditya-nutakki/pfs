from cgan import Generator, Z_DIM, num_classes, BATCH_SIZE
import torchshow as ts
import torch

device = "cuda"
model_path = "./generator.pt"

noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device = device)
ckpt = torch.load(model_path)
# print(ckpt, type(ckpt))

model = Generator().to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with torch.no_grad():
    for i in range(10):
        classes = (torch.ones(BATCH_SIZE, device=device) * i).long().to(device)
        preds = model(noise, classes)
        ts.save(preds, f"infer_{i}.jpeg")