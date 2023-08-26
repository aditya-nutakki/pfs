import os
import model_config as c
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import datasets, transforms
import time
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import cv2


"""
Use this module only as a backbone. Inherit this wherever you are using and modify methods accordingly. 
This module was written to ease testing with various blocks that should (ideally) fall into place when something is changed. 

This would work fine for JUST computer vision tasks but would need some changes if it were to be use for NLP tasks

Dataloaders (train, test and val) must be overwritten for your use case
"""


class GenericModule():
    def __init__(self, model) -> None:
        self.device = c.device
        self.nc = c.nc
        self.model = self.modify_last_layer(model)
        self.model = model.to(self.device) # expecting a torch.nn.Module

        print(model)

        self.num_epochs = c.epochs
        self.batch_size = c.batch_size
        self.input_dims = c.input_dims
        self.param_count = self.get_total_param_count(self.model)
        self.model_name = c.model_name
        self.logs_path = c.log_path
        self.mode = c.mode
        self.model_path = c.model_path
        
        print(f"Loading model in {self.mode} mode !")
        if self.mode == "infer":
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.eval()
            else:
                raise FileNotFoundError(f"Model path at {self.model_path} not found ! check 'mode' to be in either 'train' or 'infer' parameter in config file !")
        
        if os.path.exists(self.logs_path):
            os.remove(self.logs_path)
        
        print(f"Loaded Model with {self.param_count} total parameters !")

        self.opt = SGD(self.model.parameters(), lr=3e-4)
        # try optimizer with Adam
        self.loss_fn = CrossEntropyLoss()

        if self.model_name:
            self.save_dir = os.path.join(c.save_dir, self.model_name)
        else:
            self.save_dir = None

        os.makedirs(self.save_dir, exist_ok=True)
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(self.input_dims[1], self.input_dims[2])), 
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        # write custom train and test loaders since use case might be different.
        self.train_set, self.test_set = datasets.ImageFolder(c.train_path, transform=self.transforms), datasets.ImageFolder(c.test_path, transform=self.transforms)
        self.train_loader, self.test_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True), DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)
        self.class_to_idx, self.idx_to_class = self._get_class_idx_info()
        # print(self.idx_to_class)
        
    def get_total_param_count(self, model):
        # returns total number of parameters, trainable + non-trainable
        return sum(p.numel() for p in model.parameters())


    def _get_class_idx_info(self):
        class_to_idx = self.train_set.class_to_idx
        idx_to_class = {i : c for c, i in self.class_to_idx.items()}
        return class_to_idx, idx_to_class


    def infer(self, image_paths):
        images = []
        for image_path in image_paths:
            images.append(self.transforms(cv2.imread(image_path)))
            # print(images[-1].shape, type(images[-1]))
        
        images = torch.stack(images).to(self.device)
        preds = F.softmax(self.model(images), dim=1)
        conf, classes = torch.max(preds, 1)
        # print(preds, conf, classes)
        conf, classes = conf.cpu().detach().numpy(), classes.cpu().detach().numpy()
        for score, class_ in zip(conf, classes):
            print(f"Class => {self.idx_to_class[class_]}; Score => {score}")


    def modify_last_layer(self, model):
        in_num_features = model.fc.in_features
        # print(in_num_features, type(in_num_features))
        model.fc = nn.Linear(in_num_features, self.nc)
        return model


    def compute_accuracy(self, preds, labels):
        _, predicted = torch.max(preds, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy, correct


    def _write_logs(self, logs):
        # logs expected to be string
        with open(self.logs_path, "a") as f:
            f.write(logs)


    def _val(self):
        # expects model to be loaded with weights
        self.model.eval()
        correct, total_count = 0, 0
        print("Evaluating ...")
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images)
                preds = F.softmax(preds, dim=1)
                batch_acc, batch_correct = self.compute_accuracy(preds, labels)
                # print(f"batch_acc => {batch_acc}")
                correct += batch_correct
                total_count += labels.size(0)
        
        # acc = correct/len(self.test_loader)
        acc = correct/total_count
        return acc
    

    def _train(self):
        if self.mode != "train":
            raise Exception(f"Mode was set to {self.mode}; but called train function. Change Mode to 'train'")
        torch.cuda.empty_cache()
        print(f"Starting training !")

        for _e in range(self.num_epochs):
            self.model.train()
            print(f"epoch {_e+1}:")
            stime = time.time()
            running_loss, last_loss = 0.0, 0.0
            
            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.opt.zero_grad()

                preds = self.model(images)
                loss = self.loss_fn(preds, labels)
                loss.backward()
                # print(loss)
                self.opt.step()

                running_loss += loss.item()

                if i%100 == 0:
                    last_loss = running_loss/100
                    print(f"batch loss = {last_loss}")
                    # self.logs.append(f"epoch {_e+1}; batch {i}; loss {last_loss}\n")
                    self._write_logs(f"epoch {_e+1}; batch {i}; loss {last_loss}\n")
                    running_loss = 0.0

            ftime = time.time()
            print()
            print(f"Took {ftime-stime}s to complete epoch number {_e+1}")

            if self.save_dir:
                torch.save(self.model.state_dict(), self.save_dir + f"_epoch{_e+1}.pt")
            
            print("Validating on test set... ")
            acc = self._val()
            print(f"Val Acc => {acc} in epoch {_e+1}")
            self._write_logs(f"validation acc on epoch {_e+1}; val_acc {acc}\n\n")

            print()
