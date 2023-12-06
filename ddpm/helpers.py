import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, cv2
from torchvision.transforms import transforms



# custom normalizing function to get into range you want
class NormalizeToRange(nn.Module):
    def  __init__(self, min_val, max_val) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, images):
        # images could be a batch or individual
        
        # _min_val, _max_val = torch.min(images), torch.max(images)
        # return (self.max_val - self.min_val) * ((images - _min_val) / (_max_val - _min_val)) + self.min_val
        return (self.max_val - self.min_val) * ((images - 0) / (1)) + self.min_val
    


class BikesDataset(Dataset):
    def __init__(self, dataset_path, limit = -1, _transforms = None, max_len = 32, img_sz = 224) -> None:
        super().__init__()
        
        self.max_len = max_len
        self.transforms = _transforms
        
        if not self.transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_sz, img_sz)),
                # transforms.Normalize((0.5, ), (0.5,))
                NormalizeToRange(-1, 1)
            ])


        self.dataset_path = dataset_path
        # self.images_path, self.captions_path = os.path.join(self.dataset_path, "images"), os.path.join(self.dataset_path, "captions")
        # self.images, self.captions = os.listdir(self.images_path), os.listdir(self.captions_path)
        self.valid_extensions = ["jpg", "jpeg", "png", "JPEG", "JPG"]
        self.limit = limit
        self.images_path = dataset_path
        self.images = os.listdir(self.images_path)[:self.limit]

        self.images = [os.path.join(self.images_path, image) for image in self.images if image.split(".")[-1] in self.valid_extensions]
        # self.captions = [os.path.join(self.captions_path, caption) for caption in self.captions]
        # self.images = self.preprocess_images(self.images)
        # print(self.images[:20])


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # to return a pair of (images, text-caption)
        try:
            image = cv2.imread(self.images[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(image.shape)
            image = self.transforms(image)
            return image
        
        except:
            return None
        

    def preprocess_images(self, image_paths):
        clean_paths = []
        count = 0
        for index, image_path in enumerate(image_paths):
            try:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # print(image.shape)
                image = self.transforms(image)

                clean_paths.append(image_path)
            except:
                print(f"failed at {image_path}")
                count += 1

        print(f"{count} number of invalid images")

        return clean_paths


if __name__ == "__main__":
    ds = BikesDataset("/mnt/d/work/datasets/bikes/combined", img_sz=128)

