from model import CLIP
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from helpers import *

def train(clip, train_loader, test_loader = None, lr = 3e-4, epochs = 20):
    
    opt = Adam(clip.parameters(), lr = lr)
    criterion = CrossEntropyLoss()

    for i, (images, captions) in enumerate(train_loader):

        opt.zero_grad()
        
        image_embeddings = clip.image_encoder(images)
        text_embeddings = clip.text_encoder(captions)
        
        # find loss and dot product here
        loss = criterion()

        # update weights 
        opt.step()





if __name__ == "__main__":
    train_path = "/opt/infilect/aditya/datasets/tds"
    train_set = CLIPDataset(train_path)
    train_loader = DataLoader(train_set)
    clip = CLIP()
    assert clip.image_encoder.embedding_dim == clip.text_encoder.embedding_dim, "embedding dimensions should be the same for image and text encoders !"
    
    train(clip, train_loader)



