import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
