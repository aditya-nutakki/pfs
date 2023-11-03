from model import CLIP
import torch
from torchvision.transforms import transforms
from helpers import *
import torchshow as ts
import cv2
from time import time

embedding_dim, model_type = 256, "resnet"
device = "cuda"
model_path = f"./output/best_{model_type}_{embedding_dim}_15e-5.pt"


def preprocess_images(images_path):
    images = [cv2.imread(image_path) for image_path in images_path]
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    images = [_transforms(image) for image in images]
    return torch.stack([image for image in images], dim = 0)


_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5, ), (0.5,))
            ])

clip = CLIP(embedding_dim = 256, model_type = model_type)
clip.load_state_dict(torch.load(model_path))
clip.to(device)
print(f"loaded {model_type} model with {embedding_dim} dims")


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

images = [
        "/mnt/d/work/datasets/coco_captions/images/87875.jpg", # image of a hydrant
        "/mnt/d/work/datasets/coco_captions/images/85772.jpg" # image of playing tennis
        ]

captions = [
            "A fire hydrant in a grassy field next to a bush",
            "Someone swimming in the distance", # false example
            "A man is about to hit a tennis ball with a racquette."
        ]

images, captions = preprocess_images(images), tokenizer(captions, return_tensors = "pt", max_length = 64, padding = "max_length")

clip.eval()
with torch.no_grad():
    images, captions = images.to(device), captions.to(device)
    stime = time()
    image_embeddings, text_embeddings = clip(images, captions)
    logits = image_embeddings @ text_embeddings.T
    preds = F.softmax(logits, dim = -1)
    ftime = time()
    print(logits)
    print(preds)
    print(f"inference done in {ftime - stime}s")




