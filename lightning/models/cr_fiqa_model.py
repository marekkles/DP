from typing import Optional
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

class CrFiqaNet(pl.LightningModule):
    def __init__(
        self,
        backbone:str,
        backbone_args:dict,
        metric:dict,
        metric_args:dict,
        optim:str,
        optim_args:dict,
        lr_scheduler:str,
        lr_scheduler_args:dict,
        backbone_checkpoint_path:Optional[str]=None,
    ):
        super().__init__()
        assert metric == "CrFiqaLoss", "Cannot use other loss than CrFiqa"
        assert backbone in available_backbones, f"{backbone} is not valid\
             backbone, Available backbones are {' '.join(available_backbones)}"
        self.encoder = backbones.__dict__[backbone](**backbone_args)
        if not backbone_checkpoint_path is None:
            self.encoder.load_state_dict(torch.load(backbone_checkpoint_path))
        self.secondary = torch.nn.Linear(
            metric_args["in_features"], 
            1
        )
        self.decoder = metrics.CrFiqaLoss(**metric_args)
        self.optim = optim
        self.optim_args = optim_args
        self.lr_scheduler=lr_scheduler
        self.lr_scheduler_args=lr_scheduler_args
        self.train_accuracy = torchmetrics.Accuracy()
        self.validation_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.save_hyperparameters()
    def forward(self, x):
        embedding = self.encoder(x)
        secondary = self.secondary(embedding)
        return embedding, secondary
    def configure_optimizers(self):
        optimizer = torch.optim.__dict__[self.optim](
            [
                {'params': self.encoder.parameters()}, 
                {'params': self.secondary.parameters()},
                {'params': self.decoder.parameters()}
            ], 
            **self.optim_args
        )
        lr_scheduler = torch.optim.lr_scheduler.__dict__[self.lr_scheduler](
            optimizer, **self.lr_scheduler_args
        )
        return [optimizer], [lr_scheduler]
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.encoder(x)
        qs = self.secondary(z)
        loss, y_prediction = self.decoder(z, qs, y)
        self.train_accuracy(y_prediction, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_accuracy)
        return loss
    def training_epoch_end(self, outputs) -> None:
        self.train_accuracy.reset()
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.encoder(x)
        qs = self.secondary(z)
        loss, y_prediction = self.decoder(z, qs, y)
        self.validation_accuracy(y_prediction, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.validation_accuracy)
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        z = self.encoder(x)
        qs = self.secondary(z)
        loss, y_prediction = self.decoder(z, qs, y)
        self.test_accuracy(y_prediction, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.validation_accuracy)
    def predict_step(self, batch, batch_idx):
        x, y = batch
        embedding, quality = self(x)
        return {"embedding" : embedding, "quality" : quality, "label": y}