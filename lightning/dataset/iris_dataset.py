import os
import csv
import functools
from collections.abc import Callable
from typing import Optional, Tuple, Any

import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from PIL import Image

class IrisVerificationDataset(VisionDataset):
    def __init__(
        self, 
        root: str,
        autocrop: bool = True,
        transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transform=transform)
        self.autocrop = autocrop
        self.__load_subset()

    def __load_subset(self):
        self.__load_annotations_csv()
        self.__load_mapping_csv()
        self.__load_pairs_csv()
        self.__load_impostors_csv()

    def __load_annotations_csv(self):
        with open(
            os.path.join(self.root, 'annotations.csv'), newline=''
        ) as csvfile:
            csv_file = csv.reader(csvfile, delimiter=',', quotechar='"')
            self.header = dict(
                map(lambda x: (x[1], x[0]), enumerate(next(csv_file)))
            )
            self.anotations = list(csv_file)

    def __load_mapping_csv(self):
        with open(
            os.path.join(self.root, 'mapping.csv'), newline=''
        ) as csvfile:
            csv_file = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(csv_file)
            self.mapping = dict(csv_file)

    def __load_pairs_csv(self):
        with open(
            os.path.join(self.root, 'pairs.csv'), newline=''
        ) as csvfile:
            csv_file = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(csv_file)
            self.pairs = [(int(x[0]), int(x[1])) for x in csv_file]

    def __load_impostors_csv(self):
        with open(
            os.path.join(self.root, 'impostors.csv'), newline=''
        ) as csvfile:
            csv_file = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(csv_file)
            self.impostors = [(int(x[0]), int(x[1])) for x in csv_file]

    @staticmethod
    def __to_tensor(pic):
        img = torch.as_tensor(np.array(pic, copy=True))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        return img.permute((2, 0, 1))
    
    def __get_img(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index (int): Index of image file
        Returns:
            image: Loaded PIL image
        """
        annotation = self.anotations[index]
        img_path = os.path.join(
            self.root,
            self.mapping[annotation[self.header['image_id']]],
            'iris_right.UNKNOWN'
        )
        pic = Image.open(img_path).convert('L')
        if self.autocrop:
            pic = pic.crop((
                float(annotation[self.header['pos_x']]) -
                float(annotation[self.header['radius_1']]),
                float(annotation[self.header['pos_y']]) -
                float(annotation[self.header['radius_1']]),
                float(annotation[self.header['pos_x']]) +
                float(annotation[self.header['radius_1']]),
                float(annotation[self.header['pos_y']]) +
                float(annotation[self.header['radius_1']]),
            ))
        img = self.__to_tensor(pic)
        id = int(self.mapping[annotation[self.header['image_id']]])
        return img, id
    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types
        """
        img, id = self.__get_img(index)
        if self.transform is not None:
            img = self.transform(img)
        return img, id
    def __len__(self) -> int:
        return len(self.anotations)

class IrisDataset(VisionDataset):
    SUBSETS={
        'train_iris_nd_crosssensor_2013': {'size': 101968, 'num_classes': 940},
        'train_iris_casia_v4': {'size': 15890, 'num_classes': 1600},
        'train_iris_inno_a1': {'size': 79128, 'num_classes': 39659},
        'train_iris_nd_0405': {'size': 51880, 'num_classes': 570},
        'train_iris_utris_v1': {'size': 542, 'num_classes': 108},
    }
    CLASS_STRING_FORMAT='{__subset}:{subject_id}_{eye_side}'
    def __init__(
        self, 
        root: str,
        subsets: list,
        autocrop: bool = True,
        transform: Optional[Callable] = None, 
        transforms: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(
            root,
            transform=transform,
            transforms=transforms,
            target_transform=target_transform
        )
        assert type(root) == str,\
            "Parameter root can must be string"
        assert all(type(s) == str for s in subsets),\
            "Subsets must be list of strings"
        
        self.subsets = subsets
        self.autocrop = autocrop

        self.__load_subsets_anotations()
        
        assert (
            self.expected_size == self.size and
            self.expected_num_classes == self.num_classes
        ), "Expected dataset sizes do not match"
    def __load_subsets_anotations(self):
        self.entry_list=[]
        self.classes_dict={}
        for subset in self.subsets:
            self.__load_annotations(subset)
    def __load_annotations(self, subset):
        subset_path = os.path.join(self.root, subset)
        subset_annotations_path = os.path.join(subset_path, 'annotations.csv')
        assert (
            os.path.exists(subset_path) and 
            os.path.exists(subset_annotations_path)
        ), f"Subset {subset} has no annotations file or does not exist"
        with open(subset_annotations_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            header = next(reader)
            for entry in reader:
                self.__save_entry(entry, header, subset)
    def __save_entry(self, entry, header, subset):
        entry_dict=dict(zip(['__subset'] + header, [subset] + entry))
        entry_class_string = (
            IrisDataset.CLASS_STRING_FORMAT.format(**entry_dict)
        )
        if not entry_class_string in self.classes_dict:
            self.classes_dict[entry_class_string] = len(self.classes_dict)
        entry_dict['__class_string'] = entry_class_string
        entry_dict['__class_number'] = self.classes_dict[entry_class_string]
        self.entry_list.append(entry_dict)
    @property
    def size(self):
        return len(self.entry_list)
    @property
    def num_classes(self):
        return len(self.classes_dict)
    @property
    def expected_size(self):
        return sum(IrisDataset.SUBSETS[s]['size'] for s in self.subsets)
    @property
    def expected_num_classes(self):
        return sum(IrisDataset.SUBSETS[s]['num_classes'] for s in self.subsets)
    @staticmethod
    def __to_tensor(pic):
        img = torch.as_tensor(np.array(pic, copy=True))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        return img.permute((2, 0, 1))
    #@functools.lru_cache(maxsize=None)
    def __get_img(self, index: int) -> torch.Tensor:
        """
        Args:
            index (int): Index of image file
        Returns:
            image: Loaded PIL image
        """
        entry = self.entry_list[index]
        img_path = os.path.join(
            self.root, entry['__subset'],
            entry['image_id'],'iris_right.UNKNOWN'
        )
        pic = Image.open(img_path).convert('L')
        if self.autocrop:
            pic = pic.crop((
                float(entry['pos_x'])-float(entry['radius_1']),
                float(entry['pos_y'])-float(entry['radius_1']),
                float(entry['pos_x'])+float(entry['radius_1']),
                float(entry['pos_y'])+float(entry['radius_1']),
            ))
        return self.__to_tensor(pic)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types
        """
        img = self.__get_img(index)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.entry_list[index]['__class_number']
    def __len__(self) -> int:
        return self.size



class DatasetSubset(VisionDataset):
    def __init__(
        self, 
        dataset: VisionDataset,
        indicies: list,
        transform: Optional[Callable]=None,
    ) -> None:
        super().__init__(dataset.root, transform=transform)
        self.dataset=dataset
        self.transform=transform
        self.indicies=indicies
        self.num_classes=dataset.num_classes
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        idx = self.indicies[index]
        img, target = self.dataset[idx]
        if self.transform != None:
           img = self.transform(img)
        return img, target
    def __len__(self) -> int:
        return len(self.indicies)