import numpy as np
import os
import pytorch_lightning as pl
import random
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config import update_args, parse_arguments
from data.dataset import set_dataloader
from models.select_model import select_model


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    checkpoint_callback = ModelCheckpoint(monitor='validation_loss', mode='min')

    train_dataloader = set_dataloader(args, phase='train')
    val_dataloader = set_dataloader(args, phase='val')
    test_dataloader = set_dataloader(args, phase='test')

    model = select_model(args, device)

    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), name='lightning_log')

    trainer = pl.Trainer(
                    callbacks=[checkpoint_callback],
                    gpus=args.TRAIN.NUM_GPUS,
                    logger=tb_logger,
                    max_epochs=args.TRAIN.NUM_EPOCHS
                    )
    
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(ckpt_path=None, test_dataloaders=test_dataloader)

    os.makedirs('./weights', exist_ok=True)
    save_path = f'./weights/{args.TRAIN.RUN_NAME}.ckpt'
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    option = parse_arguments()
    args = update_args(cfg_file=option.cfg_file, run_name=option.run_name, seed=option.seed)
    fix_seed(args.TRAIN.SEED)
    main(args)
    print("-------------------------")
    print("     Finish All")
    print("-------------------------")
