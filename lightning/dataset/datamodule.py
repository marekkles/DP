from .dataset import IrisVerificationDataset, IrisDataset
import torch
from torch.utils.data import random_split, DataLoader
from typing import Optional

import pytorch_lightning as pl

class IrisDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: list,
            predic_data_dir: str,
            auto_crop: bool = True,
            batch_size: int = 32,
            train_transforms = None,
            predict_transforms = None,
        ):
        super().__init__()
        self.train_transforms = train_transforms
        self.predict_transforms = predict_transforms
        self.data_dir = data_dir
        self.predict_data_dir = predic_data_dir
        self.auto_crop = auto_crop
        self.batch_size = batch_size
        self.iris_full = IrisDataset(
            self.data_dir, 
            transform=self.train_transforms, 
            autocrop=self.auto_crop
        )
        self.num_classes = self.iris_full.num_classes
    def setup(self, stage: Optional[str] = None):
        self.iris_predict = IrisVerificationDataset(
            self.predict_data_dir, 
            transform=self.predict_transforms, 
            autocrop=self.auto_crop
        )
        traint_val_test_split = [
            int(len(self.iris_full) * 0.75), 
            int(len(self.iris_full) * 0.2), 
            int(len(self.iris_full) * 0.05)
        ]
        traint_val_test_split[-1] += len(self.iris_full) - sum(traint_val_test_split)
        self.iris_train, self.iris_val, self.iris_test  = random_split(
            self.iris_full, 
            traint_val_test_split, 
            generator=torch.Generator().manual_seed(42)
        )
    def train_dataloader(self):
        return DataLoader(self.iris_train, batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.iris_val, batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.iris_test, batch_size=self.batch_size)
    def predict_dataloader(self):
        return DataLoader(self.iris_predict, batch_size=self.batch_size)

