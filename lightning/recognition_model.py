from torchvision import transforms
import torch
import torch.nn as nn
import pytorch_lightning as pl
import backbones
import metrics
import torchmetrics

available_backbones = list(
    filter(
        lambda x: 
            x[0] != '_' and 
            not x[0].isupper() and 
            not x[1].isupper() and 
            x[-1].isdigit(),
        backbones.__dict__
    )
)

available_metrics = list(
    filter(
        lambda x: x[0] != '_' and x[0].isupper(),
        metrics.__dict__
    )
)

class RecognitionNet(pl.LightningModule):
    def __init__(
        self,
        backbone:str,
        backbone_args:dict,
        metric:str,
        metric_args:dict
    ):

        super().__init__()
        assert backbone in available_backbones, f"{backbone} is not valid\
             backbone, Available backbones are {' '.join(available_backbones)}"
        assert metric in available_metrics, f"{metric} is not valid\
             metric, Available metrics are {' '.join(available_metrics)}"

        self.encoder = backbones.__dict__[backbone](**backbone_args)
        self.decoder = metrics.__dict__[metric](**metric_args)
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.validation_accuracy = torchmetrics.Accuracy()
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.encoder(x)
        y_prediction = self.decoder(z, y)
        self.train_accuracy(y_prediction, y)
        loss = self.loss(y_prediction, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_accuracy)
        return loss
    def training_epoch_end(self, outputs) -> None:
        self.log('train_acc_epoch', self.train_accuracy)
        self.train_accuracy.reset()
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.encoder(x)
        y_prediction = self.decoder(z, y)
        self.validation_accuracy(y_prediction, y)
        loss = self.loss(y_prediction, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.validation_accuracy)
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        z = self.encoder(x)
        y_prediction = self.decoder(z)
        loss = self.loss(y_prediction, y)
        self.log('test_loss', loss)
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x), y

