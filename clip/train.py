from model import CLIP
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from helpers import *
import torchshow as ts

def train(clip, train_loader, test_loader = None, lr = 3e-4, epochs = 20):
    
    opt = Adam(clip.parameters(), lr = lr)
    criterion = CrossEntropyLoss()

    for i, (images, captions) in enumerate(train_loader):
        # we have to assume that 
        batch_size = images.shape[0]

        opt.zero_grad()
        # ts.save(images, f"batch_{i}.jpeg")
        # print(captions.shape)
        image_embeddings, text_embeddings = clip(images, captions)
        targets = torch.arange(0, batch_size)
        predictions = image_embeddings @ text_embeddings.T
        # print("something something")
        # find loss and dot product here
        # image_loss, text_loss = criterion(), criterion()
        loss = criterion(predictions, targets)
        loss.backward()
        print(loss)
        # update weights 
        opt.step()
        # break




if __name__ == "__main__":
    train_path = "/mnt/d/work/datasets/coco_captions"
    train_set = CLIPDataset(train_path)
    train_loader = DataLoader(train_set, batch_size = 2, shuffle = True)
    clip = CLIP()
    # assert clip.image_encoder.embedding_dim == clip.text_encoder.embedding_dim, "embedding dimensions should be the same for image and text encoders !"
    
    train(clip, train_loader)



