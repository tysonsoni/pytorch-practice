import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


def to_tensor(sample):
    inputs, targets = sample
    return torch.from_numpy(inputs), torch.from_numpy(targets)


dataset = WineDataset(transform=to_tensor)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
