import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2

s = 7 # divide image into s x s grid
b = 2 # number of bounding boxes
nc = 4 # number of classes
img_dims = (3, 448, 448) # c, h, w

lambda_coord, lambda_noobj = 5, 0.5

class YOLOLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, preds, target):
        # preds and target to be in the shape of (_, s, s, (5*b) + c)
        loss = 0
        return loss


class TrafficDataset(Dataset):
    def __init__(self, base_path = "/mnt/d/work/datasets/traffic_set/ts", txt_path = "../train.txt") -> None:
        super().__init__()
        self.base_path = base_path
        self.txt_path = txt_path # path to be relative to base_path
        self.images, self.annots = self._get_img_annots()
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(img_dims[1], img_dims[2])),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.len_per_cell = 1/s

        self.mapping = {
            0: "prohibitory",
            1: "danger",
            2: "mandatory",
            3: "other"
        }


    def _get_img_annots(self):
        txt_file = os.path.join(self.base_path, self.txt_path)

        with open(txt_file, "r") as f:
            files = [l.strip().replace("\n", "") for l in f.readlines()]

        files = [os.path.join(self.base_path, f.split("/")[-1]) for f in files]

        images, annots = [], []

        for file in files:
            if file.endswith(".jpg"):
                images.append(file)
                annots.append(os.path.join(self.base_path, file.replace(".jpg", ".txt")))

        assert len(images) == len(annots)
        # print(images[:10])
        # print()
        # print(annots[:10])
        return images, annots


    def _cvt_annot(self, annots):
        # converts str to float
        labels = []
        for annot in annots:
            _class, x, y, w, h = annot.split(" ")
            _class, x, y, w, h = float(_class), float(x), float(y), float(w), float(h)
            labels.append([_class, x, y, w, h])
        return labels


    def _create_gt(self, annotation):
        with open(annotation, "r") as f:
            _labels = [l.strip().replace("\n", "") for l in f.readlines() if l]
        _labels = self._cvt_annot(_labels)
        print(_labels)
        # labels = torch.zeros((s, s, b*5 + nc))
        labels = torch.zeros((s, s, 5 + nc))
        
        for l, _label in enumerate(_labels):
            class_, xc, yc, w, h = _label[0], _label[1], _label[2], _label[3], _label[4]

            x_cell, y_cell = int(xc/self.len_per_cell), int(yc/self.len_per_cell)
            new_x, new_y = (xc - x_cell*self.len_per_cell)/self.len_per_cell, (yc - y_cell*self.len_per_cell)/self.len_per_cell # calculating relative distance from the grid

            labels[x_cell, y_cell, :5] = torch.tensor([1.0, new_x, new_y, w, h]) # is of shape [Pc, x, y, w, h, c1, c2 ... cn]
            labels[x_cell, y_cell, 5 + int(class_)] = torch.tensor(1.0)

        return labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        annotation = self.annots[index]
        # both are just paths, need to do some preproessing before returning this
        # image = self.transforms(Image.open(image))
        image = self.transforms(cv2.imread(image))
        annotation = self._create_gt(annotation)

        return image, annotation
        # return super().__getitem__(index)
    

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
    # x = torch.randn(img_dims).unsqueeze(dim = 0)
    # model = Model()
    # y = model(x)
    # print(y.shape)
    ds = TrafficDataset()
    # print(ds[1])
    img, ann = ds[3]
    print(img.shape, ann.shape)
    print(ann)