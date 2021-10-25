import torch.nn as nn

from pytorch_lightning import LightningModule


class ConvBatchNormRelu(LightningModule):
    def __init__(self, input_channel, kernel_size, output_channel, padding):
        super(ConvBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
