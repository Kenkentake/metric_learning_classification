import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule

from models.utils_model import ConvBatchNormRelu
from utils import save_umap

class ArcfaceModel(LightningModule):
    """
    Args:
        in_features: number of dimensions of feature vector
        out_features: number of class
        s: scale factor (= 1 / temperature for softmax)
        m: margin (radian)
        easy_margin: dealing with margin problem (when theta > pi - m)
    """
    def __init__(self, args, device):
        super(ArcfaceModel, self).__init__()
        self.args = args
        self._device = device
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

        self.weight = nn.Parameter(torch.FloatTensor(254, 1024))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, label):
        cnn_output = self.feature_extractor_cnn(x)
        fcl_input = cnn_output.view(-1, 256 * 2 * 2)
        # arcface part
        # l2 normalize x and W
        cos = F.linear(F.normalize(fcl_input), F.normalize(self.weight))
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
        return output

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
        outputs = self(inputs, labels)
        loss = self.cross_entropy_loss(outputs, labels) 
        return {
            'count': labels.shape[0],
            'loss': loss
        }
    
    def training_epoch_end(self, outputs):
        count = 0
        cross_entropy_loss = 0.0
        for output in outputs:
            count += output['count']
            cross_entropy_loss += output['loss'].data.item()

        training_epoch_outputs = {
            'training_cross_entropy_loss': cross_entropy_loss / count
        }
        self.logger.log_metrics(training_epoch_outputs, step=self.current_epoch)
        return None

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs, labels)
        loss = self.cross_entropy_loss(outputs, labels) 
        return {
            'count': labels.shape[0],
            'loss': loss
        }
    
    def validation_epoch_end(self, outputs):
        count = 0
        cross_entropy_loss = 0.0
        for output in outputs:
            count += output['count']
            cross_entropy_loss += output['loss'].data.item()

        validation_epoch_outputs = {
            'validation_cross_entropy_loss': cross_entropy_loss / count
        }
        self.logger.log_metrics(validation_epoch_outputs, step=self.current_epoch)
        return None

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs, labels)
        loss = self.cross_entropy_loss(outputs, labels) 
        return {
            'count': labels.shape[0],
            'embeddings': outputs,
            'labels': labels,
            'loss': loss
        }
    
    def test_epoch_end(self, outputs):
        embeddings_all = []
        labels_all = []
        count = 0
        cross_entropy_loss = 0.0
        
        for output in outputs:
            count += output['count']
            cross_entropy_loss += output['loss'].data.item()
            embeddings_all.append(output['embeddings'].cpu())
            labels_all.append(output['labels'].cpu())

        test_epoch_outputs = {
            'test_cross_entropy_loss': cross_entropy_loss / count
        }
        fig_umap = save_umap(np.concatenate(embeddings_all), np.concatenate(labels_all), self.args.TRAIN.SEED)
        self.logger.experiment.add_figure("Arcface UMAP", fig_umap)
        self.logger.log_metrics(test_epoch_outputs, step=self.current_epoch)

        return None
