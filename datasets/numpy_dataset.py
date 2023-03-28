# Pytorch dataset for numpy arrays
########################################################################################################################
# Import dependencies
import torch
from torch.utils.data import Dataset
import torchio as tio
import numpy as np
########################################################################################################################

class numpy_dataset(Dataset):  # Inherit from Dataset class
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


