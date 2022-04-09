import os
import pickle
from .iris_dataset import verification_dataset_factory, IrisDataset, DatasetSubset
import torch
from torch import randperm
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple

import pytorch_lightning as pl

__all__ = ['IrisDataModule']

class IrisDataModule(pl.LightningDataModule):
    def __init__(
            self,
            root_dir: str,
            batch_size: int,
            num_workers: int,
            num_in_channels: int,
            train_pseudolabels: Optional[str] = None,
            train_subset: Optional[list] = None,
            auto_crop: Optional[bool] = True,
            unwrap: Optional[bool] = False,
            shuffle: Optional[bool] = True,
            train_transform: Optional[torch.nn.Module] = None,
            val_transform: Optional[torch.nn.Module] = None,
            test_transform: Optional[torch.nn.Module] = None,
            predict_transform: Optional[torch.nn.Module] = None,
            traint_val_test_split: Optional[Tuple[float, float, float]] = (0.75,0.2,0.05)
        ):
        super().__init__()
        assert len(traint_val_test_split) == 3, "Train val test split tuple must contain 3 elements"
        assert all(map(lambda x: type(x) == float, traint_val_test_split)), "Train val test split tuple must be tuple of floats between 0 and 1"
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_in_channels = num_in_channels
        
        if not train_pseudolabels is None:
            with open(os.path.join(self.root_dir, train_pseudolabels)) as f:
                self.train_pseudolabels = pickle.load(f)
        else:
            self.train_pseudolabels = None

        if train_subset is None:
            self.train_subset = self.list_train_sets(self.root_dir)
        else:
            self.train_subset = tuple(train_subset)

        self.auto_crop = auto_crop
        self.unwrap = unwrap
        self.shuffle = shuffle
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.predict_transform = predict_transform
        self.traint_val_test_split = traint_val_test_split


        self.iris_full = IrisDataset(
            self.root_dir,
            self.num_in_channels,
            self.train_subset,
            pseudolabels=self.train_pseudolabels,
            autocrop=self.auto_crop,
            unwrap=self.unwrap
        )

        self.iris_predict = dict([
            (
                d,
                verification_dataset_factory(
                    os.path.join(self.root_dir, d),
                    self.num_in_channels,
                    transform=self.predict_transform,
                    autocrop=self.auto_crop,
                    unwrap=self.unwrap
                )
            ) for d in self.list_verification_sets(self.root_dir)
        ])

        self.iris_predict["iris_verification_pseudo"] = verification_dataset_factory(
            self.root_dir,
            self.num_in_channels,
            subset=self.train_subset,
            transform=self.predict_transform,
            autocrop=self.auto_crop,
            unwrap=self.unwrap,
        )

        self.iris_predict_selector = list(self.iris_predict.keys())[0]
    def iris_predict_select(self, key):
        assert key in self.iris_predict_list, 'Selected dataset is not in predict list'
        self.iris_predict_selector = key
    @property
    def iris_predict_list(self):
        return tuple(self.iris_predict.keys())
    @staticmethod
    def list_train_sets(root_dir: str) -> List[str]:
        return list(filter(
            lambda s: s.startswith('train_iris_') and os.path.isdir(
                os.path.join(root_dir, s)
            ), os.listdir(root_dir)
        ))
    @staticmethod
    def list_verification_sets(root_dir: str) -> List[str]:
        return list(filter(
            lambda s: s.startswith('iris_verification_') and os.path.isdir(
                os.path.join(root_dir, s)
            ), os.listdir(root_dir)
        ))
    @property
    def num_classes(self):
        return self.iris_full.num_classes
    def setup(self, stage: Optional[str] = None):
        lengths = [int(l*len(self.iris_full)) for l in  self.traint_val_test_split]
        lengths[-1] += len(self.iris_full) - sum(lengths)
        if self.shuffle:
            perms = randperm(sum(lengths)).tolist()
        else:
            perms = torch.arange(0, sum(lengths), 1, dtype=torch.int32).tolist()
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
            self.iris_predict[self.iris_predict_selector],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )

