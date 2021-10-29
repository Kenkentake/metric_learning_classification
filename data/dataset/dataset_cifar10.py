import numpy as np
import torch

from torchvision.datasets import CIFAR10


class CIFAR10Dataset:
    def __init__(self, class_list, phase, root_path, train_size_rate, transform):
        self.prepare(class_list, phase, root_path, train_size_rate, transform)

    def prepare(self, class_list, phase, root_path, train_size_rate, transform):
        _train_val_dataset = CIFAR10(download=True, root=root_path, train=True, transform=transform)
        test_dataset = CIFAR10(download=True, root=root_path, train=False, transform=transform)

        train_val_dataset = self.extract_data(_train_val_dataset, class_list)

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
        elif phase == 'test':
            self.dataset = test_dataset
        return None

    def extract_data(self, dataset, class_list):
        index_list = [i for i, label in enumerate(dataset.targets) if label in class_list]
        dataset.data = dataset.data[index_list, :]
        dataset.targets = np.array(dataset.targets)[index_list].tolist()
        return dataset
