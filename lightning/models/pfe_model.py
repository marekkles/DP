from torchvision import transforms
import torch
import torch.nn as nn
import pytorch_lightning as pl
import backbones
import metrics
import torchmetrics
from metrics import UncertaintyHead, MLSLoss

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

class PfeNet(pl.LightningModule):
    def __init__(
        self,
        backbone_checkpoint_path:str,
        backbone:str,
        backbone_args:dict,
        optim:str,
        optim_args:dict,
        lr_scheduler:str,
        lr_scheduler_args:dict
    ):
        super().__init__()
        assert backbone in available_backbones, f"{backbone} is not valid\
             backbone, Available backbones are {' '.join(available_backbones)}"
        self.encoder = backbones.__dict__[backbone](**backbone_args)
        self.encoder.load_state_dict(torch.load(backbone_checkpoint_path))

        self.uncertainty_head = UncertaintyHead(backbone_args["num_classes"])
        self.loss = MLSLoss()
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
        deviation = self.uncertainty_head(embedding)
        return embedding, deviation
    def configure_optimizers(self):
        optimizer = torch.optim.__dict__[self.optim](
            [
                {'params': self.encoder.parameters()},
                {'params': self.uncertainty_head.parameters()},
            ], 
            **self.optim_args
        )
        lr_scheduler = torch.optim.lr_scheduler.__dict__[self.lr_scheduler](
            optimizer, **self.lr_scheduler_args
        )
        return [optimizer], [lr_scheduler]
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        embedding, deviation = self(x)
        loss = self.loss(embedding, deviation, y)
        self.log('train_loss', loss)
        return loss
    def training_epoch_end(self, outputs) -> None:
        self.train_accuracy.reset()
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        embedding, deviation = self(x)
        loss = self.loss(embedding, deviation, y)
        self.log('val_loss', loss)
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        embedding, deviation = self(x)
        loss = self.loss(embedding, deviation, y)
        self.log('test_loss', loss)
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x), y
