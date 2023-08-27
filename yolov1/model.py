import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
import os


s = 7 # divide image into s x s grid
b = 2 # number of bounding boxes
nc = 20 # number of classes
img_dims = (3, 448, 448) # c, h, w

lambda_coord, lambda_noobj = 5, 0.5




class YOLOLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, preds, target):
        # preds and target to be in the shape of (_, s, s, (5*b) + c)
        loss = 0
        return loss



class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv7x7 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=7, stride=2, padding = 3)
        self.maxpool = nn.MaxPool2d(2,stride=2)

        self.conv3x3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride = 1, padding =1)
        
        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding = 1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding = 1)
        )
        
        self.module2 = nn.Sequential(
            nn.Conv2d(in_channels= 512, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels= 256, out_channels=512, kernel_size=3, padding = 1),

            nn.Conv2d(in_channels= 512, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels= 256, out_channels=512, kernel_size=3, padding = 1),

            nn.Conv2d(in_channels= 512, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels= 256, out_channels=512, kernel_size=3, padding = 1),

            nn.Conv2d(in_channels= 512, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels= 256, out_channels=512, kernel_size=3, padding = 1),

            nn.Conv2d(in_channels= 512, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels= 512, out_channels=1024, kernel_size=3, padding = 1),
        )

        self.module3 = nn.Sequential(
            nn.Conv2d(in_channels= 1024, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels= 512, out_channels=1024, kernel_size=3, padding = 1),
            nn.Conv2d(in_channels= 1024, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels= 512, out_channels=1024, kernel_size=3, padding = 1),
            nn.Conv2d(in_channels= 1024, out_channels=1024, kernel_size=3, padding = 1),
            nn.Conv2d(in_channels= 1024, out_channels=1024, kernel_size=3, stride=2, padding = 1)
        )

        self.module4 = nn.Sequential(
            nn.Conv2d(in_channels= 1024, out_channels=1024, kernel_size=3, padding = 1),
            nn.Conv2d(in_channels= 1024, out_channels=1024, kernel_size=3, padding = 1)
        )

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024*7*7, 4096)
        self.linear2 = nn.Linear(4096, s*s*(b*5 + nc))



    def forward(self, x):
        x = F.leaky_relu(self.conv7x7(x))
        x = self.maxpool(x)
        x = F.leaky_relu(self.conv3x3(x))
        x = self.maxpool(x)

        x = F.leaky_relu(self.module1(x))
        x = self.maxpool(x)

        x = F.leaky_relu(self.module2(x))
        x = self.maxpool(x)

        x = F.leaky_relu(self.module3(x))
        x = F.leaky_relu(self.module4(x))

        x = self.flatten(x)
        x = F.leaky_relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        x = x.view((s,s,-1))

        return x




if __name__ == "__main__":
    x = torch.randn(img_dims).unsqueeze(dim = 0)
    model = Model()
    y = model(x)
    print(y.shape)