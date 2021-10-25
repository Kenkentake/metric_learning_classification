import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.reducers import ThresholdReducer

from models.utils_model import ConvBatchNormRelu
from utils import save_umap

class TripletNetModel(LightningModule):
    def __init__(self, args, device):
        super(TripletNetModel, self).__init__()
        self.args = args
        self._device = device
        self.learning_rate = args.TRAIN.LEARNING_RATE
        self.margin = args.TRAIN.MARGIN

        self.distance = CosineSimilarity()
        # reducer: receive all losses for each pair and calculate the final loss
        self.reducer = ThresholdReducer(low = 0)
        self.triplet_loss = TripletMarginLoss(margin=self.margin, distance=self.distance, reducer=self.reducer)
        # miner: make pairs of triplet
        self.miner = TripletMarginMiner(margin=self.margin, distance=self.distance)
        self.cross_entropy_loss = nn.CrossEntropyLoss

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

        self.feature_extractor_fcl = nn.Linear(256 * 2 * 2, 256)

    def forward(self, x):
        cnn_output = self.feature_extractor_cnn(x)
        fcl_input = cnn_output.view(-1, 256 * 2 * 2)
        output = self.feature_extractor_fcl(fcl_input) 
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
        embeddings = self(inputs)
        triplets = self.miner(embeddings, labels)
        loss = self.triplet_loss(embeddings, labels, triplets) 
        return {
            'count': labels.shape[0],
            'loss': loss
        }
    
    def training_epoch_end(self, outputs):
        count = 0
        triplet_loss = 0.0
        for output in outputs:
            count += output['count']
            triplet_loss += output['loss'].data.item()

        training_epoch_outputs = {
            'training_triplet_loss': triplet_loss / count
        }
        self.logger.log_metrics(training_epoch_outputs, step=self.current_epoch)
        return None

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        embeddings = self(inputs)
        triplets = self.miner(embeddings, labels)
        loss = self.triplet_loss(embeddings, labels, triplets) 
        return {
            'count': labels.shape[0],
            'loss': loss
        }
    
    def validation_epoch_end(self, outputs):
        count = 0
        triplet_loss = 0.0
        for output in outputs:
            count += output['count']
            triplet_loss += output['loss'].data.item()

        validation_epoch_outputs = {
            'validation_triplet_loss': triplet_loss / count
        }
        self.logger.log_metrics(validation_epoch_outputs, step=self.current_epoch)
        return None

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        embeddings = self(inputs)
        triplets = self.miner(embeddings, labels)
        loss = self.triplet_loss(embeddings, labels, triplets) 
        return {
            'count': labels.shape[0],
            'embeddings': embeddings,
            'labels': labels,
            'loss': loss
        }
    
    def test_epoch_end(self, outputs):
        embeddings_all = []
        labels_all = []
        count = 0
        triplet_loss = 0.0
        
        for output in outputs:
            count += output['count']
            triplet_loss += output['loss'].data.item()
            embeddings_all.append(output['embeddings'].cpu())
            labels_all.append(output['labels'].cpu())

        test_epoch_outputs = {
            'test_triplet_loss': triplet_loss / count
        }
        fig_umap = save_umap(np.concatenate(embeddings_all), np.concatenate(labels_all), self.args.TRAIN.SEED)
        self.logger.experiment.add_figure("Triplet UMAP", fig_umap)
        self.logger.log_metrics(test_epoch_outputs, step=self.current_epoch)

        return None
