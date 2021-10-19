import numpy as np
import torch

from torchvision.datasets import CIFAR10


class CIFAR10Dataset:
    def __init__(self, phase, root_path, train_size_rate, transform):
        self.prepare(phase, root_path, train_size_rate, transform)

    def prepare(self, phase, root_path, train_size_rate, transform):
        train_val_dataset = CIFAR10(download=True, root_path, train=True, transform=transform)
        test_dataset = CIFAR10(download=True, root_path, train=False, transform=transform)
        # split to train and validation
        train_size = int(len(train_val_dataset) * train_size_rate)
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
                                            train_val_dataset,
                                            [train_size, val_size]
                                            )
        
        if phase == 'train':
            self.dataset = train_dataset
        elif phase == 'val':
            self.dataset = val_dataset
        else phase == 'test'
            self.dataset = test_dataset
        return None
