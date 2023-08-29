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
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

s = 7 # divide image into s x s grid
b = 2 # number of bounding boxes
nc = 4 # number of classes
img_dims = (3, 448, 448) # c, h, w

lambda_coord, lambda_noobj = 5, 0.5


save_dir = "./output/"
os.makedirs(save_dir, exist_ok=True)


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
        # print(bboxes, bboxes.shape)
        vals, idx = torch.max(bboxes[:, 0], dim = 0)
        return bboxes[idx], objectness_scores
    

    def forward(self, preds, targets):
        # preds and target to be in the shape of (-1, s, s, (5*b) + c)
        obj_loss, no_obj_loss = 0, 0
        
        for k, (pred, target) in enumerate(zip(preds, targets)):
            s = pred.shape[0]
            
            for i in range(s):
                for j in range(s):
                    # print(f"bboxes are => {pred[i, j, :b*5]}, shape => {pred[i, j, :b*5].shape}")
                    best_bbox, objectness_scores = self._get_bbox(pred[i, j, :b*5])
                    best_bbox, objectness_scores = best_bbox.to("cuda"), objectness_scores.to("cuda")
                    # print(f"best_bbox => {best_bbox}")
                    # c_p, x_p, y_p, w_p, h_p = best_bbox.tolist()
                    # print(target[i, j, :].tolist())
                    c = target[i, j, :].tolist()[0]
                    target_format_pred = torch.cat([best_bbox, pred[i, j, b*5:]])
                    # print(f"normal pred => {target_format_pred}; shape => {target_format_pred.shape}, target_shape => {target[i,j, :]}; shape=> {target[i,j,:].shape}")

                    if c == 1:
                        # ground truth has object
                        # box_loss = F.mse_loss(best_bbox[1], target[i, j, 1]) + F.mse_loss(best_bbox[2], target[i, j, 2]) + F.mse_loss(torch.sqrt(best_bbox[3]), torch.sqrt(target[i, j, 3])) + F.mse_loss(torch.sqrt(best_bbox[4]), torch.sqrt(target[i, j, 4]))
                        box_loss = F.mse_loss(best_bbox[1], target[i, j, 1]) + F.mse_loss(best_bbox[2], target[i, j, 2]) + F.mse_loss(best_bbox[3], target[i, j, 3]) + F.mse_loss(best_bbox[4], target[i, j, 4])
                        pc_loss = F.mse_loss(best_bbox[0], target[i, j, 0])
                        # print(target_format_pred[5:], target_format_pred[5:].shape)
                        # print(target[i, j, 5:], target[i, j, 5:].shape)
                        class_loss = F.mse_loss(target_format_pred[5:], target[i, j, 5:])
                        obj_loss += lambda_coord*box_loss + pc_loss + class_loss

                    else:
                        # ground truth does not have object
                        no_obj_loss += lambda_noobj*torch.sum(objectness_scores**2) # since we would be subtracting with 0 and then squaring it, its the same as squaring and summing it

                loss = obj_loss + no_obj_loss
        # print(loss)
        return loss


class YOLOLossV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def mse_loss(self, input, target):
        return torch.sum((input - target) ** 2)

    def weighted_mse_loss(self, input, target, weight):
        return torch.sum(weight * (input - target) ** 2)

    def forward(self, preds, targets):
        return 

class TrafficDataset(Dataset):
    def __init__(self, base_path = "/mnt/d/work/datasets/traffic_set/ts", txt_path = "../train.txt") -> None:
        super().__init__()
        self.base_path = base_path
        self.txt_path = txt_path # path to be relative to base_path
        self.images, self.annots = self._get_img_annots()
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(img_dims[1], img_dims[2]))
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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
        print(f"getting annotations from {annotation}")
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
        self.linear1 = nn.Linear(1024*7*7, 1024) # using 1024 instead of 4096 because model cant train on my 3060 notebook GPU (6GB)
        self.dropout = nn.Dropout(0.25)
        self.linear2 = nn.Linear(1024, s*s*(b*5 + nc))



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
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        
        x = x.view((-1, s, s, 5*b + nc))

        return x


def train(model, train_dataloader):
    device = "cuda"
    model = model.to(device)
    print(f"training on a model with {sum(p.numel() for p in model.parameters())} parameters") # 112,240,366 params
    criterion = YOLOLoss()
    opt = torch.optim.Adam(model.parameters(), lr = 10e-6)

    # cosine_lr = CosineAnnealingLR(opt, 100*300)


    print("Starting to train !")
    torch.cuda.empty_cache()
    min_loss = 9999

    for e in range(120):
        stime = time.time()
        losses = []
        model_save_path = os.path.join(save_dir, f"model_{e}.pt")
        print(f"Training on epoch {e} ...")
        model.train()
        for i, (image, label) in enumerate(train_dataloader):
          
            opt.zero_grad()
            image, label = image.to(device), label.to(device)

            preds = model(image)
            loss = criterion(preds, label)
            losses.append(loss.item())
            loss.backward()
            opt.step()

            if i%25 == 0:
                print(f"loss on epoch {e}; step {i} => {loss}")
                print(torch.count_nonzero(preds))
        
        ftime = time.time()
        avg_loss = sum(losses)/len(losses)
        # print(f"loss => {sum(losses)}; len => {len(losses)}")
        print(f"Avg epoch loss at epoch {e} => {round(avg_loss, 4)}; time taken => {round(ftime-stime, 2)}s")
        
        # if avg_loss <= min_loss:
        #     min_loss = avg_loss
        #     print(f"saving model at {model_save_path} ...")  
        #     torch.save(model.state_dict(), model_save_path)
        #     print(f"saved for epoch {e}")
        print()    

def test_loss(labels):
    crit = YOLOLoss()
    preds = torch.rand(3, s, s, 5*b + nc).to("cuda") # output
    labels = labels.to("cuda")
    loss = crit(preds, labels)
    return loss


def get_x1y1x2y2(xc, yc, w, h):
    return int(xc-w/2), int(yc-h/2), int(xc+w/2), int(yc+h/2)



def vis(image, xc, yc, w, h, i, j):
    image = image.permute(1, 2, 0)
    image = image.detach().cpu().numpy()
    w, h = int(w * img_dims[1]), int(h * img_dims[2])
    xc = int(img_dims[1]*((i+xc)/s))
    yc = int(img_dims[2]*((j+yc)/s))
    # print(xc, yc, w, h)
    x1, y1, x2, y2 = get_x1y1x2y2(xc, yc, w, h)
    # print(x1, y1, x2, y2)
    image = image*255
    image = np.ascontiguousarray(image, dtype=np.uint8)
    print(image.shape, type(image))
    # image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 255), thickness=2)
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 0), 2)
    cv2.imshow("vis", image)
    cv2.waitKey(0)



def decode_preds(images, targets):
    # targets to be in the shape of (-1, s, s 5*b + nc)
    found = False
    for image, target in zip(images, targets):
        for i in range(s):
            for j in range(s):
                cell = target[i, j, :5]
                pc, xc, yc, w, h = cell
                if pc > 0.01:
                    print("visualising ...")
                    vis(image, xc, yc, w, h, i, j)
                    


if __name__ == "__main__":
    # x = torch.randn(4, 3, 448, 448)
    device = "cuda"
    model = YOLO().to(device)
    # y = model(x)
    
    # print(y.shape)
    load_model = False

    if load_model:
        model.load_state_dict(torch.load("./output/model_old.pt"))
        model.to(device)
        print(f"loaded pretrained model")

    train_ds = TrafficDataset()
    train_loader = DataLoader(train_ds, batch_size = 4, shuffle=True)
    # 2744 total number of elements at output
    # train(model, train_loader)
    for images, targets in train_loader:
        images = images.to(device)
        preds = model(images)
        decode_preds(images, preds)
        break
