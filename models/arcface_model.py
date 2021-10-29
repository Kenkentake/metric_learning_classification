import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule

from models.utils_model import ConvBatchNormRelu
from utils import save_confusion_matrix, save_umap

class ArcfaceModel(LightningModule):
    """
    Args:
        s: scale factor (= 1 / temperature for softmax)
        m: margin (radian)
        easy_margin: dealing with margin problem (when theta > pi - m)
    """
    def __init__(self, args, device):
        super(ArcfaceModel, self).__init__()
        self.args = args
        self._device = device
        self.output_dim = args.TRAIN.OUTPUT_DIM
        self.learning_rate = args.TRAIN.LEARNING_RATE
        self.scale_factor = self.args.TRAIN.SCALE_FACTOR
        self.margin = self.args.TRAIN.MARGIN
        self.easy_margin = self.args.TRAIN.EASY_MARGIN
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
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
        if self.output_dim == 10:
            self.fc2 = nn.Linear(128, 10)
            self.weight = nn.Parameter(torch.FloatTensor(10, 10))
        elif self.output_dim == 8:
            self.fc2 = nn.Linear(128, 8)
            self.weight = nn.Parameter(torch.FloatTensor(8, 8))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, label=None):
        cnn_output = self.feature_extractor_cnn(x)
        fcl_input = cnn_output.view(-1, 256 * 2 * 2)
        embeddings = self.fc1(fcl_input)
        x = self.relu(embeddings)
        x = self.fc2(x)
        
        if label is None:
            return embeddings, x
        else:
            # arcface part
            # l2 normalize x and W
            cos = F.linear(F.normalize(x), F.normalize(self.weight))
            # angular margin penalty
            sin = torch.sqrt((1.0 - torch.pow(cos, 2)).clamp(0, 1))
            # phi: cos(theta + m)
            phi = cos * self.cos_m - sin * self.sin_m
            if self.easy_margin:
                phi = torch.where(cos > 0, phi, cos)
            else:
                phi = torch.where(cos > self.th, phi, cos - seif.mm) 
            one_hot = torch.zeros(cos.size(), device=self._device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cos)
            output *= self.scale_factor
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
        embeddings, outputs = self(inputs)
        if self.output_dim == 10:
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
        elif self.output_dim == 8:
            return {
                'embeddings': outputs,
                'labels': labels,
            }
    
    def test_epoch_end(self, outputs):
        embeddings_all = []
        labels_all = []
            
        if self.output_dim == 10:
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
            fig_conf_matrix, _ = save_confusion_matrix(labels_conf_matrix, preds_conf_matrix)
            self.logger.experiment.add_figure("Arcface Confusion Matrix", fig_conf_matrix)
            fig_umap = save_umap(np.concatenate(embeddings_all), np.concatenate(labels_all), self.args.TRAIN.SEED)
            self.logger.experiment.add_figure("Arcface UMAP", fig_umap)
            test_epoch_outputs = {
                'test_accuracy': accuracy / count,
                'test_cross_entropy_loss': cross_entropy_loss / count
            }
            self.logger.log_metrics(test_epoch_outputs, step=self.current_epoch)

        elif self.output_dim == 8:
            for output in outputs:
                embeddings_all.append(output['embeddings'].cpu())
                labels_all.append(output['labels'].cpu())

        fig_umap = save_umap(np.concatenate(embeddings_all), np.concatenate(labels_all), self.args.TRAIN.SEED)
        self.logger.experiment.add_figure("Arcface UMAP", fig_umap)
        return None
