import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule

from models.utils_model import ConvBatchNormRelu
from utils import save_confusion_matrix, save_umap

class CNNModel(LightningModule):
    def __init__(self, args, device):
        super(CNNModel, self).__init__()
        self.args = args
        self._device = device
        self.learning_rate = args.TRAIN.LEARNING_RATE
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.feature_extractor_cnn = nn.Sequential(
                            ConvBatchNormRelu(3, 3, 32, 1),
                            nn.MaxPool2d(2, 2),
                            ConvBatchNormRelu(32, 3, 64, 1),
                            nn.MaxPool2d(2, 2),
                            nn.Dropout(0.3),
                            ConvBatchNormRelu(64, 3, 128, 1),
                            nn.MaxPool2d(2, 2),
                            ConvBatchNormRelu(128, 3, 256, 1),
                            nn.MaxPool2d(2, 2),
                            nn.Dropout(0.3)) 

        self.fc1 = nn.Linear(256 * 2 * 2, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 10)
        
        self.weight = nn.Parameter(torch.FloatTensor(10, 10))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, label):
        cnn_output = self.feature_extractor_cnn(x)
        fcl_input = cnn_output.view(-1, 256 * 2 * 2)
        embeddings = self.fc1(fcl_input)
        x = self.relu(embeddings)
        output = self.fc2(x)
        return embeddings, output

    def configure_optimizers(self):
        if self.args.TRAIN.OPTIMIZER_TYPE == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, 
                                  momentum=self.args.TRAIN.MOMENTUM)
        if self.args.TRAIN.OPTIMIZER_TYPE == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        _, outputs = self(inputs, labels)
        loss = self.cross_entropy_loss(outputs, labels) 
        accuracy = (outputs.argmax(1) == labels).sum().item()
        return {
            'accuracy': accuracy,
            'count': labels.shape[0],
            'loss': loss
        }
    
    def training_epoch_end(self, outputs):
        accuracy = cross_entropy_loss = 0.0
        count = 0
        for output in outputs:
            accuracy += output['accuracy']
            count += output['count']
            cross_entropy_loss += output['loss'].data.item()

        training_epoch_outputs = {
            'training_accuracy': accuracy / count,
            'training_cross_entropy_loss': cross_entropy_loss / count
        }
        self.logger.log_metrics(training_epoch_outputs, step=self.current_epoch)
        return None

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        _, outputs = self(inputs, labels)
        loss = self.cross_entropy_loss(outputs, labels) 
        accuracy = (outputs.argmax(1) == labels).sum().item()
        return {
            'accuracy': accuracy,
            'count': labels.shape[0],
            'loss': loss
        }
    
    def validation_epoch_end(self, outputs):
        accuracy = cross_entropy_loss = 0.0
        count = 0
        for output in outputs:
            accuracy += output['accuracy']
            count += output['count']
            cross_entropy_loss += output['loss'].data.item()

        validation_epoch_outputs = {
            'validation_accuracy': accuracy / count,
            'validation_cross_entropy_loss': cross_entropy_loss / count
        }
        self.logger.log_metrics(validation_epoch_outputs, step=self.current_epoch)
        return None

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        embeddings, outputs = self(inputs, labels)
        loss = self.cross_entropy_loss(outputs, labels) 
        accuracy = (outputs.argmax(1) == labels).sum().item()
        return {
            'preds': outputs.argmax(1),
            'accuracy': accuracy,
            'count': labels.shape[0],
            'embeddings': outputs,
            'labels': labels,
            'loss': loss
        }
    
    def test_epoch_end(self, outputs):
        embeddings_all = []
        labels_all = []
        preds_all = []
        labels_conf_matrix = []
        preds_conf_matrix = []
        accuracy = cross_entropy_loss = 0.0
        count = 0
        
        for output in outputs:
            preds_conf_matrix.extend(output['preds'].tolist())
            labels_conf_matrix.extend(output['labels'].tolist())
            accuracy += output['accuracy']
            count += output['count']
            cross_entropy_loss += output['loss'].data.item()
            embeddings_all.append(output['embeddings'].cpu())
            labels_all.append(output['labels'].cpu())
            preds_all.append(output['preds'].cpu())

        fig_conf_matrix = save_confusion_matrix(labels_conf_matrix, preds_conf_matrix)
        self.logger.experiment.add_figure("cnn Confusion Matrix", fig_conf_matrix)

        test_epoch_outputs = {
            'test_accuracy': accuracy / count,
            'test_cross_entropy_loss': cross_entropy_loss / count
        }
        fig_umap = save_umap(np.concatenate(embeddings_all), np.concatenate(labels_all), self.args.TRAIN.SEED)
        self.logger.experiment.add_figure("CNN UMAP", fig_umap)
        self.logger.log_metrics(test_epoch_outputs, step=self.current_epoch)

        return None
