import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ClassificationModel(pl.LightningModule):
    def __init__(self, optimizer, learning_rate, max_epochs, batch_size):
        super(ClassificationModel, self).__init__()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size

    def forward(self, X):
        pass

    def training_step(self, X):
        pass

    def validation_step(self, X):
        pass

    def configure_optimizer(self):
        pass
