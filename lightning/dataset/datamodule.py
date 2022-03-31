from .iris_dataset import IrisVerificationDataset, IrisDataset, DatasetSubset
import torch
from torch import randperm
from torch.utils.data import DataLoader
from typing import Optional

import pytorch_lightning as pl

class IrisDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            subsets: list,
            predic_data_dir: str,
            auto_crop: bool = True,
            unwrap: bool = False,
            shuffle: bool = True,
            batch_size: int = 32,
            num_workers: int = 4,
            train_transform = None,
            val_transform = None,
            test_transform = None,
            predict_transform = None,
            traint_val_test_split = (0.75,0.2,0.05)
        ):
        super().__init__()
        assert len(traint_val_test_split) == 3, "Train val test split tuple must contain 3 elements"
        assert all(map(lambda x: type(x) == float, traint_val_test_split)), "Train val test split tuple must be tuple of floats between 0 and 1"
        self.data_dir = data_dir
        self.predict_data_dir = predic_data_dir
        self.auto_crop = auto_crop
        self.unwrap = unwrap
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.predict_transform = predict_transform
        self.traint_val_test_split = traint_val_test_split
        self.iris_full = IrisDataset(
            self.data_dir,
            subsets,
            autocrop=self.auto_crop,
            unwrap=self.unwrap
        )
        self.iris_predict = IrisVerificationDataset(
            self.predict_data_dir,
            transform=self.predict_transform,
            autocrop=self.auto_crop,
            unwrap=self.unwrap
        )
    @property
    def num_classes(self):
        return self.iris_full.num_classes
    def setup(self, stage: Optional[str] = None):
        lengths = [int(l*len(self.iris_full)) for l in  self.traint_val_test_split]
        lengths[-1] += len(self.iris_full) - sum(lengths)
        if self.shuffle:
            perms = randperm(sum(lengths)).tolist()
        else:
            perms = torch.range(0, sum(lengths), dtype=torch.int32).tolist()
        aggs = [sum(lengths[:0]), sum(lengths[:1]), sum(lengths[:2]), sum(lengths[:3])]
        subsets = [perms[start:end] for start, end in zip(aggs[:-1], aggs[1:])]
        
        self.iris_train = DatasetSubset(self.iris_full, subsets[0], self.train_transform)
        self.iris_val = DatasetSubset(self.iris_full, subsets[1], self.val_transform)
        self.iris_test = DatasetSubset(self.iris_full, subsets[2], self.test_transform)
    def train_dataloader(self):
        return DataLoader(
            self.iris_train, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.iris_val, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    def test_dataloader(self):
        return DataLoader(
            self.iris_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    def predict_dataloader(self):
        return DataLoader(
            self.iris_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )

