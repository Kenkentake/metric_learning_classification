import numpy as np
import torch

from torchvision.datasets import CIFAR10


class CIFAR10Dataset:
    def __init__(self, phase, root_path,, transform):
        self.prepare(phase, root_path, transform)

    def prepare(self, phase, root_path, transform):
        train_dataset = CIFAR10(download=True, root_path, train=True, transform=transform)
        test_dataset = CIFAR10(download=True, root_path, train=False, transform=transform)

        if phase == 'train':
            self.dataset = train_dataset
        if phase == 'test'
            self.dataset = test_dataset
        return None
