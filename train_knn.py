import numpy as np
import os
import pytorch_lightning as pl
import random
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.tensorboard import SummaryWriter

from config import update_args, parse_arguments
from data.dataset import set_dataloader
from models.select_model import select_model
from utils import save_confusion_matrix


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    train_dataloader = set_dataloader(args, phase='train')
    val_dataloader = set_dataloader(args, phase='val')
    test_dataloader = set_dataloader(args, phase='test')

    model = select_model(args, device)
    model.load_state_dict(torch.load(args.TRAIN.WEIGHT_PATH))

    train_embeddings = []
    train_labels = []
    test_embeddings = []
    test_labels = []

    for (inputs, labels) in train_dataloader:
        embeddings = model(inputs)
        train_embeddings.append(embeddings.to('cpu').detach().numpy())
        train_labels.append(labels.to('cpu').detach().numpy())

    for (inputs, labels) in test_dataloader:
        embeddings = model(inputs)
        test_embeddings.append(embeddings.to('cpu').detach().numpy())
        test_labels.append(labels.to('cpu').detach().numpy())
        
    train_x = np.concatenate(train_embeddings)
    train_y = np.concatenate(train_labels)
    test_x = np.concatenate(train_embeddings)
    test_y = np.concatenate(train_labels)

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(train_x, train_y)
    pred_y = knn_clf.predict(test_x)

    fig_conf_matrix = save_confusion_matrix(test_y, pred_y)
    log_path = f'./lightning_log/{args.TRAIN.RUN_NAME}/'
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)
    writer.add_figure(f'{args.TRAIN.RUN_NAME} Confusion Matrix', fig_conf_matrix)
    writer.close()


if __name__ == '__main__':
    option = parse_arguments()
    args = update_args(cfg_file=option.cfg_file, run_name=option.run_name, seed=option.seed)
    fix_seed(args.TRAIN.SEED)
    main(args)
    print("-------------------------")
    print("     Finish All")
    print("-------------------------")
