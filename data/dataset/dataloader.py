import torch

from torch.utils.data import DataLoader
from data.dataset.dataset_cifar10 import CIFAR10Dataset
from data.dataset.transforms import set_transforms


def set_dataloader(args, phase):
    batch_size = args.TRAIN.BATCH_SIZE
    num_workers = args.DATA.NUM_WORKERS
    root_path = args.DATA.ROOT_PATH
    transform = set_transforms(args.DATA.TRANSFORM_LIST, args.DATA.IMG_SIZE)

    dataset = CIFAR10Dataset(phase, root_path, transform).dataset
    dataloader = DataLoader(dataset, batch_size=batchsize,
                            num_workers=num_workers, sampler=None, shuffle=True)
    return dataloader
