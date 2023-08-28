import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from utils import *

s = 7 # divide image into s x s grid
b = 2 # number of bounding boxes
nc = 4 # number of classes
img_dims = (3, 448, 448) # c, h, w

lambda_coord, lambda_noobj = 5, 0.5

class YOLOLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _get_bbox(self, pred):
        # pred contains elements only up to the b*5 elements

        # if you want to count in conditional probability, then argmax(class_probabilities) * Pc -> return this max value's bounding box
        bboxes = []
        objectness_scores = []
        for i in range(b):
            bboxes.append(pred[i*5 : (i + 1)*5].tolist())
            objectness_scores.append(pred[i*5].tolist())
        # print(f"_get_bbox boxes => {bboxes}, len => {len(bboxes)}")
        bboxes = torch.tensor(bboxes)
        objectness_scores = torch.tensor(objectness_scores)
        print(bboxes, bboxes.shape)
        vals, idx = torch.max(bboxes[:, 0], dim = 0)
        return bboxes[idx], objectness_scores
    

    


    def forward(self, preds, targets):
        # preds and target to be in the shape of (-1, s, s, (5*b) + c)
        obj_loss, noobj_loss = 0, 0
        
        for k, (pred, target) in enumerate(zip(preds, targets)):
            s = pred.shape[0]
            
            for i in range(s):
                for j in range(s):
                    # print(f"bboxes are => {pred[i, j, :b*5]}, shape => {pred[i, j, :b*5].shape}")
                    best_bbox, objectness_scores = self._get_bbox(pred[i, j, :b*5])
                    # print(f"best_bbox => {best_bbox}")
                    c_p, x_p, y_p, w_p, h_p = best_bbox.tolist()
                    c, x, y, w, h = target[i, j, :].tolist()
                    target_format_pred = torch.cat([best_bbox, pred[i, j, b*5:]])
                    # print(f"normal pred => {target_format_pred}; shape => {target_format_pred.shape}, target_shape => {target[i,j, :]}; shape=> {target[i,j,:].shape}")

                    if c == 1:
                        # ground truth has object
                        box_loss = F.mse_loss(best_bbox[1], target[1]) + F.mse_loss(best_bbox[2], target[2]) + F.mse_loss(torch.sqrt(best_bbox[3]), torch.sqrt(target[3])) + F.mse_loss(torch.sqrt(best_bbox[4]), torch.sqrt(target[4]))
                        pc_loss = F.mse_loss(best_bbox[0], target[0])
                        class_loss = F.mse_loss(target_format_pred[i, j, 5:], target[i, j, 5:])
                        obj_loss += lambda_coord*class_loss + pc_loss + box_loss

                    else:
                        # ground truth does not have object
                        no_obj_loss += lambda_noobj*torch.sum(objectness_scores**2) # since we would be subtracting with 0 and then squaring it, its the same as squaring and summing it

                    break
                break
                loss = obj_loss + noobj_loss

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
        # print(_labels)
        # labels = torch.zeros((s, s, b*5 + nc))
        labels = torch.zeros((s, s, 5 + nc))
        
        for l, _label in enumerate(_labels):
            class_, xc, yc, w, h = _label[0], _label[1], _label[2], _label[3], _label[4]

            x_cell, y_cell = int(xc/self.len_per_cell), int(yc/self.len_per_cell)
            new_x, new_y = (xc - x_cell*self.len_per_cell)/self.len_per_cell, (yc - y_cell*self.len_per_cell)/self.len_per_cell # calculating relative distance from the grid

            labels[x_cell, y_cell, :5] = torch.tensor([1.0, new_x, new_y, w, h]) # is of shape [Pc, x, y, w, h]
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
    

class YOLO(nn.Module):
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
        
        x = x.view((-1, s, s, 5*b + nc))

        return x




def train(model, train_dataloader):
    device = "cuda"
    model = model.to(device)
    criterion = YoloLoss()
    opt = torch.optim.Adam(model.parameters(), lr = 10e-4)


    for e in range(10):
        print(f"Training on epoch {e} ...")
        for i, (image, label) in enumerate(train_dataloader):
            model.train()

            opt.zero_grad()
            image, label = image.to(device), label.to(device)

            preds = model(image)
            loss = criterion(preds, label)
            loss.backward()
            opt.step()

            if i%50 == 0:
                print(f"loss on epoch {e}; step {i} => {loss}")


def test_loss(labels):
    crit = YOLOLoss()
    preds = torch.randn(4, s, s, 5*b + nc) # output
    loss = crit(preds, labels)



if __name__ == "__main__":
    # x = torch.randn(img_dims).unsqueeze(dim = 0)
    # model = YOLO()
    # y = model(x)
    # print(y.shape)

    train_ds = TrafficDataset()
    # _, annot = train_ds[3]
    train_loader = DataLoader(train_ds, batch_size = 4, shuffle=True)
    for _, labels in train_loader:
        test_loss(labels)
        break
    # train(model, train_loader)