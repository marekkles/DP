from torchvision import transforms
import torch
import torch.nn as nn
import pytorch_lightning as pl
import backbones
import metrics
import torchmetrics
from typing import List, Tuple, Optional

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
available_losses = ['L1Loss', 'SmoothL1Loss', 'MSELoss']

class SddFiqaNet(pl.LightningModule):
    def __init__(
        self,
        backbone:str,
        backbone_args:dict,
        loss:str,
        loss_args:dict,
        optim:str,
        optim_args:dict,
        lr_scheduler:str,
        lr_scheduler_args:dict,
        backbone_checkpoint_path:Optional[str]=None,
    ):
        super().__init__()
        assert backbone in available_backbones, f"{backbone} is not valid\
             backbone, Available backbones are {' '.join(available_backbones)}"
        assert loss in available_losses, f"{loss} is not valid\
             loss, Available losses are {' '.join(available_losses)}"
        self.encoder = backbones.__dict__[backbone](
            **backbone_args, num_classes=1
        )
        if backbone_checkpoint_path is not None:
            self.encoder.load_state_dict(torch.load(backbone_checkpoint_path))
        self.loss = torch.nn.__dict__[loss](**(loss_args if loss_args != None else {}))
        self.optim = optim
        self.optim_args = optim_args
        self.lr_scheduler=lr_scheduler
        self.lr_scheduler_args=lr_scheduler_args
        self.save_hyperparameters()
    def forward(self, x):
        quality = self.encoder(x)
        return quality
    def configure_optimizers(self):
        optimizer = torch.optim.__dict__[self.optim](
            [
                {'params': self.encoder.parameters()}
            ], 
            **self.optim_args
        )
        lr_scheduler = torch.optim.lr_scheduler.__dict__[self.lr_scheduler](
            optimizer, **self.lr_scheduler_args
        )
        return [optimizer], [lr_scheduler]
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        quality = self(x)
        loss = self.loss(quality[:,0], y)
        self.log('train_loss', loss)
        return loss
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        quality = self(x)
        loss = self.loss(quality[:,0], y)
        self.log('val_loss', loss)
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        quality = self(x)
        loss = self.loss(quality[:,0], y)
        self.log('test_loss', loss)
    def predict_step(self, batch, batch_idx):
        x, y = batch
        quality = self(x)
        return  {"quality" : quality, "label": y}
