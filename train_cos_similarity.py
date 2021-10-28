import numpy as np
import os
import pytorch_lightning as pl
import random
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
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

    model_name = args.TRAIN.MODEL_TYPE
    model = select_model(args, device)
    model.load_state_dict(torch.load(args.TRAIN.WEIGHT_PATH))

    train_embeddings = []
    train_labels = []
    test_embeddings = []
    test_labels = []

    for (inputs, labels) in train_dataloader:
        if model_name == 'triplet_net':
            embeddings = model(inputs)
        else:
            embeddings, _ = model(inputs)
        train_embeddings.append(embeddings.to('cpu').detach().numpy())
        train_labels.append(labels.to('cpu').detach().numpy())

    for (inputs, labels) in test_dataloader:
        if model_name == 'triplet_net':
            embeddings = model(inputs)
        else:
            embeddings, _ = model(inputs)
        test_embeddings.append(embeddings.to('cpu').detach().numpy())
        test_labels.append(labels.to('cpu').detach().numpy())
        
    train_x = np.concatenate(train_embeddings)
    train_y = np.concatenate(train_labels)
    test_x = np.concatenate(test_embeddings)
    test_y = np.concatenate(test_labels)

    # calculate cosine similarity
    cos_similarities = np.dot(test_x, train_x.T) / (np.linalg.norm(test_x) * np.linalg.norm(train_x, axis=1))
    most_similarity_indices = np.argmax(cos_similarities, axis=1)
    get_preds = np.vectorize(lambda x: train_y[x], otypes=[np.ndarray])
    pred_y = get_preds(most_similarity_indices)

    print(f'test_y shape is {test_y.shape}')
    print(f'pred_y shape is {pred_y.shape}')
    print(type(pred_y[0]))
    print(np.unique(test_y))
    print(np.unique(pred_y))
    print(pred_y)
    print(test_y)

    fig_conf_matrix = save_confusion_matrix(test_y.tolist(), pred_y.tolist())
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
