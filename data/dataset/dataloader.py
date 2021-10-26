import torch

from torch.utils.data import DataLoader
from data.dataset.dataset_cifar10 import CIFAR10Dataset
from data.dataset.transforms import set_transforms


def set_dataloader(args, phase):
    batch_size = args.TRAIN.BATCH_SIZE
    class_list = args.DATA.CLASS_LIST
    num_workers = args.DATA.NUM_WORKERS
    root_path = args.DATA.ROOT_PATH
    train_size_rate = args.DATA.TRAIN_SIZE_RATE
    transform = set_transforms(args.DATA.TRANSFORM_LIST, args.DATA.IMG_SIZE)

    dataset = CIFAR10Dataset(class_list, phase, root_path, train_size_rate, transform).dataset

    if phase == 'train':
        shuffle = True
    else:
        shuffle = False
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, sampler=None, shuffle=shuffle)
    return dataloader
