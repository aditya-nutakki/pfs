from model import CLIP
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from helpers import *
import torchshow as ts
import os
from time import time
import numpy as np
from transformers import DistilBertTokenizer

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"saved model to {save_path}")

save_dir = "./output"
embedding_dim = 256
os.makedirs(save_dir, exist_ok = True)



def clip_loss(image_embeddings, text_embeddings, logit_scale = 1):
    bs = image_embeddings.shape[0] # batch_size
    targets = torch.arange(bs, device = device)
    logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
    logits_per_text = logit_scale * text_embeddings @ image_embeddings.T

    loss = (F.cross_entropy(logits_per_image, targets) + F.cross_entropy(logits_per_text, targets))/2
    return loss

class ClipLoss(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * (image_features @ text_features.T)
        logits_per_text = logit_scale * (text_features @ image_features.T)
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale = 1.0, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        # return {"contrastive_loss": total_loss}
        return total_loss



def train(clip, train_loader, test_loader = None, lr = 2e-4, epochs = 30):
    
    opt = Adam(clip.parameters(), lr = lr)
    # image_criterion, text_criterion = CrossEntropyLoss(), CrossEntropyLoss()
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print(f"training on {sum([p.numel() for p in clip.parameters()])} parameters")
    min_loss = 99999
    logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07)))
    clip_fn = ClipLoss()
    torch.cuda.empty_cache()

    k = 0.5
    ep = 0
    # for ep in range(epochs):
    while min_loss >= k:
        losses = []
        stime = time()
        for i, (images, captions) in enumerate(train_loader):
            batch_size = images.shape[0]
            images, captions = images.to(device), captions.to(device)
            
            opt.zero_grad()
            # ts.save(images, f"batch_{i}.jpeg")
            # print(tokenizer.batch_decode(captions["input_ids"], skip_special_tokens = True))
            # print()        
            image_embeddings, text_embeddings = clip(images, captions)
    
            # normalizing tensors
            image_embeddings, text_embeddings = image_embeddings/ torch.sqrt(torch.sum(image_embeddings**2, dim=-1)).view(-1, 1), text_embeddings / torch.sqrt(torch.sum(text_embeddings**2, dim=-1)).view(-1, 1)
            # l2 normalize a vector by b = a / torch.sqrt(torch.sum(a**2, dim=-1)).view(-1, 1)

            loss = clip_fn(image_embeddings, text_embeddings, logit_scale)
            loss.backward()
            
            if i % 20 == 0:
                print(f"loss => {loss}; step => {i}; epoch => {ep}")
                # print(predictions, targets)
            
            losses.append(loss)
            # update weights 
            opt.step()
        

        ftime = time()
        epoch_loss = sum(losses)/len(losses)
        
        print(f"End of epoch {ep}; loss => {epoch_loss}; took {ftime - stime}s")
        if epoch_loss < min_loss:
            # best epoch
            min_loss = epoch_loss
            save_model(model = clip, save_path = os.path.join(save_dir, f"best_{clip.model_type}_{embedding_dim}.pt"))

        ep += 1
        print()
        


if __name__ == "__main__":
    device = "cuda"
    train_path = "/mnt/d/work/datasets/coco_captions/trial"
    train_set = CLIPDataset(train_path)
    train_loader = DataLoader(train_set, batch_size = 18, shuffle = True, num_workers = 3)
    clip = CLIP(embedding_dim = embedding_dim, model_type = "resnet").to(device)
    # assert clip.image_encoder.embedding_dim == clip.text_encoder.embedding_dim, "embedding dimensions should be the same for image and text encoders !"
    
    train(clip, train_loader)



