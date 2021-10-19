import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule
from pytorch_metric_learning.distances import CosineSimilarit
from pytorch_metric_learning.losses import TripletMarginLos
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.reducers import ThresholdReducer

from utils import save_confusion_matrix

class TripletNetModel(LightningModule):
    def __init__(self, args, device):
        super(TripletNetModel, self).__init__()
        self.args = args
        self.new_device = device
        self.learning_rate = args.TRAIN.LEARNING_RATE

        self.distance = Cosineimilarity()
        # reducer: receive all losses for each pair and calculate the final loss
        self.reducer = ThresholdReducer(low = 0)
        self.triplet_loss = TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
        # miner: make pairs of triplet
        self.miner = TripletMarginMiner(margin=0.2, distance=distance)
        self.cross_entropy_loss = nn.CrossEntropyLoss

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Conv2d(6, 16, 5)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
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
            'loss': triplet_loss
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

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        embeddings = self(inputs)
        triplets = self.miner(embeddings, labels)
        loss = self.triplet_loss(embeddings, labels, triplets) 
        return {
            'count': labels.shape[0],
            'loss': triplet_loss
        }
    
    def test_epoch_end(self, outputs):
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
