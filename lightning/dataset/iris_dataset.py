from ast import Dict
import os
import csv
import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
import random
from typing import List, Optional, Tuple, Any, Union
import io
import math

import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from PIL import Image

__all__ = [
    'IrisDatasetBase', 'IrisVerificationDataset', 
    'IrisVerificationDatasetV2', 'IrisVerificationDatasetV1', 
    'IrisVerificationDatasetPseudo', 'IrisDataset', 'DatasetSubset'
]

class IrisDatasetBase(VisionDataset):
    """A abstract base class for iris dataset objects
    It is necessary to override the ``size``, ``get_img_annotation``,
    ``get_img_label`` and ``get_img_binary`` method.
    Args:
        VisionDataset: Base class
    """
    def __init__(
        self,
        root: str,
        autocrop: bool,
        unwrap: bool,
        transform: torch.nn.Module,
    ) -> None:
        """Initialize base class 

        Args:
            root (str): Root directory for dataset
            autocrop (bool): Automaticaly crop to the center of iris
            unwrap (bool): Automaticaly unwrap iris into polar coordinates
            transform (torch.Module): Transforms to be applied
        """
        super(IrisDatasetBase, self).__init__(root, transform=transform)
        self.autocrop = autocrop
        self.unwrap = unwrap
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # #   METHODS TO OVERRIDE   # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    @property
    def size(self) -> int:
        """Get size of the dataset

        Raises:
            NotImplementedError: Needs to be implemented in derived class

        Returns:
            int: Size of the dataset
        """
        raise NotImplementedError
    def get_img_annotation(self, index: int) -> dict:
        """Get annotation dictionary for selected image index

        Args:
            index (int): Index of entry

        Raises:
            NotImplementedError: Needs to be implemented in derived class

        Returns:
            dict: Dictionary containing key-value entries for selected image
        """
        raise NotImplementedError
    def get_img_label(self, index: int) -> Any:
        """Get label for selected image index

        Args:
            index (int): Index of entry

        Raises:
            NotImplementedError: Needs to be implemented in derived class

        Returns:
            Any: Given label
        """
        raise NotImplementedError
    def get_img_binary(self, index: int) -> bytes:
        """Get binary of input file for entry in selected index

        Args:
            index (int): Index of entry

        Raises:
            NotImplementedError: Needs to be implemented in derived class

        Returns:
            bytes: Representation of the file in form of bytes.
        """
        raise NotImplementedError
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # #   PREIMPLEMENTED METHODS  # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def get_img(self, index: int) -> Tuple[torch.Tensor]:
        """Get tensor of image on the specified index

        Args:
            index (int): Image index 

        Returns:
            Tuple[torch.Tensor]: Image tensor auto-cropped if selected and also unwrapped
        """
        annotation = self.get_img_annotation(index)
        img_binary = self.get_img_binary(index)
        pic = Image.open(io.BytesIO(img_binary)).convert('L')
        if self.autocrop:
            pic = self.__crop(
                pic, 
                float(annotation['pos_x']),
                float(annotation['pos_y']),
                float(annotation['radius_1']),
            )
        img = self.__to_tensor(pic)
        if self.unwrap:
            assert self.autocrop, "Needs autocrop to unwrap"
            img = self.__unwrap(img, float(annotation['radius_1']))
        return img
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """Indexing in dataset. Returns a tuple of input tensor and a label.

        Args:
            index (int): Index of the image

        Returns:
            Tuple[torch.Tensor, Any]: Tuple for training containing 
            transformed image and a label
        """
        img = self.get_img(index)
        label = self.get_img_label(index)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self) -> int:
        """Get length of the dataset

        Returns:
            int: Length
        """
        return self.size
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # #  STATIC METHODS   # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    @staticmethod
    def __to_tensor(pic: Image) -> torch.Tensor:
        """Transfrom input image to tensor withowt any warnings

        Args:
            pic (Image): Input picture

        Returns:
            torch.Tensor: Image tensor of (C, W, H), where C is the number of 
            channels, W is the width and H is the height
        """
        img = torch.as_tensor(np.array(pic, copy=True))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        return img.permute((2, 0, 1))
    @staticmethod
    def __unwrap(img: torch.Tensor, radius_1: float) -> torch.Tensor:
        """Unwrap iris image into polar coordinates

        Args:
            img (torch.Tensor): Input auto-cropped image
            radius_1 (float): Radius of the iris

        Returns:
            torch.Tensor: Tensor of image transformed into polar coordinates
            with size of (1, radius_1, radius_1*pi)
        """
        iris_radius = radius_1
        nsamples = int(radius_1*math.pi)
        theta = torch.linspace(0, 2 * math.pi, nsamples)[:-1][None,:]
        radius = torch.linspace(0, iris_radius, int(iris_radius))[:-1][None,:]
        x = (radius.T @ torch.sin(theta)[None,:] + iris_radius).flatten().long()
        y = (radius.T @ torch.cos(theta)[None,:] + iris_radius).flatten().long()
        return img[0][x,y].reshape((radius.shape[1], theta.shape[1]))[None,:]
    @staticmethod
    def __crop(pic: Image, pos_x: float, pos_y: float, radius_1: float) -> Image:
        """Crop image to position and size of the iris 

        Args:
            pic (Image): Original image imported from a file
            pos_x (float): X position of the center of the iris
            pos_y (float): Y position of the center of the iris
            radius_1 (float): Radius of iris

        Returns:
            Image: Cropped image
        """
        return pic.crop((
            pos_x - radius_1,
            pos_y - radius_1,
            pos_x + radius_1,
            pos_y + radius_1,
        ))

class IrisVerificationDataset(IrisDatasetBase):
    def __init__(
        self, 
        root: str,
        autocrop: Optional[bool] = True,
        unwrap: Optional[bool] = False,
        transform: Optional[Callable] = None,
        train_subset: Optional[List[str]] = None
    ) -> None:
        """Verification dataset which combines v1 and v2 ways of loading data

        Args:
            root (str): Root directory of the dataset
            autocrop (bool, optional): Autocrop image. Defaults to True.
            unwrap (bool, optional): Unwrap iris. Defaults to False.
            transform (Callable, optional): Optional transforms. Defaults to None.
            train_sets (List[str], optional): Specify pseudo verification set.

        Raises:
            ValueError: If root directory does not contain valid structure.
        """
        super(IrisVerificationDataset, self).__init__(
            root, autocrop, unwrap, transform
        )
        self.dataset: IrisDatasetBase = None
        if train_subset is None:
            if self.is_v1(root):
                self.dataset = IrisVerificationDatasetV1(
                    root, autocrop, unwrap, transform
                )
            elif self.is_v2(root):
                self.dataset = IrisVerificationDatasetV2(
                    root, autocrop, unwrap, transform
                )
            else:
                raise ValueError
        else:
            self.dataset = IrisVerificationDatasetPseudo(
                root, train_subset, autocrop, unwrap, transform
            )
    @staticmethod
    def is_v1(root: str):
        val = True
        val = val and os.path.exists(os.path.join(root, 'annotations.csv'))
        val = val and os.path.exists(os.path.join(root, 'mapping.csv'))
        val = val and os.path.exists(os.path.join(root, 'pairs.csv'))
        val = val and os.path.exists(os.path.join(root, 'impostors.csv'))
        return val
    @staticmethod
    def is_v2(root: str):
        val = True
        val = val and os.path.exists(os.path.join(root, 'labels.csv'))
        val = val and os.path.exists(os.path.join(root, 'pairs.csv'))
        val = val and os.path.exists(os.path.join(root, 'images'))
        return val
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # #  OVERRIDDEN METHODS   # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    @property
    def size(self):
        return self.dataset.size
    @property
    def pairs(self):
        return self.dataset.pairs
    @property
    def impostors(self):
        return self.dataset.impostors
    def get_img_annotation(self, index: int) -> dict:
        return self.dataset.get_img_annotation(index)
    def get_img_label(self, index: int) -> str:
        return self.dataset.get_img_label(index)
    def get_img_binary(self, index: int) -> bytes:
        return self.dataset.get_img_binary(index)


class IrisVerificationDatasetV2(IrisDatasetBase):
    def __init__(
        self, 
        root: str,
        autocrop: bool = True,
        unwrap: bool = False,
        transform: Optional[Callable] = None
    ) -> None:
        super(IrisVerificationDatasetV2, self).__init__(
            root, autocrop, unwrap, transform
        )
        self.__load_subset()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # #  OVERRIDDEN METHODS   # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    @property
    def size(self):
        return len(self.anotations)
    def get_img_annotation(self, index: int) -> dict:
        return self.anotations[index]
    def get_img_label(self, index: int) -> int:
        return self.get_img_annotation(index)['image_id']
    def get_img_binary(self, index: int) -> bytes:
        annotation = self.get_img_annotation(index)
        img_path = os.path.join(
            self.root,'images',
            annotation['image_id']+".png"
        )
        with open(img_path, "rb") as i:
            return i.read()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # #  HELPER METHODS   # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
    def __load_subset(self):
        self.__load_labels_csv()
        self.__load_pairs_csv()

    def __load_labels_csv(self):
        with open(
            os.path.join(self.root, 'labels.csv'), newline=''
        ) as csvfile:
            csv_file = csv.reader(csvfile, delimiter=';', quotechar='"')
            self.header = next(csv_file)
            self.anotations = [
                dict(zip(self.header, entry)) for entry in csv_file
            ]

    def __load_pairs_csv(self):
        with open(
            os.path.join(self.root, 'pairs.csv'), newline=''
        ) as csvfile:
            csv_file = csv.reader(csvfile, delimiter=';', quotechar='"')
            next(csv_file)
            self.raw = [(x[0], x[1], int(x[2])) for x in csv_file]
            self.impostors = [
                (x[0], x[1]) for x in filter(
                    lambda x: x[2] == 0, self.raw
                )
            ]
            self.pairs = [
                (x[0], x[1]) for x in filter(
                    lambda x: x[2] == 1, self.raw
                )
            ]

class IrisVerificationDatasetV1(IrisDatasetBase):
    def __init__(
        self, 
        root: str,
        autocrop: bool = True,
        unwrap: bool = False,
        transform: Optional[Callable] = None
    ) -> None:
        super(IrisVerificationDatasetV1, self).__init__(
            root, autocrop, unwrap, transform
        )
        self.__load_subset()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # #  OVERRIDDEN METHODS   # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    @property
    def size(self):
        return len(self.anotations)
    def get_img_annotation(self, index: int) -> dict:
        return self.anotations[index]
    def get_img_label(self, index: int) -> int:
        annotation = self.get_img_annotation(index)
        return int(self.mapping[annotation['image_id']])
    def get_img_binary(self, index: int) -> bytes:
        annotation = self.get_img_annotation(index)
        img_path = os.path.join(
            self.root,
            self.mapping[annotation['image_id']],
            'iris_right.UNKNOWN'
        )
        with open(img_path, "rb") as i:
            return i.read()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # #  HELPER METHODS   # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
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
            self.header = next(csv_file)
            self.anotations = [
                dict(zip(self.header, entry)) for entry in csv_file
            ]

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
    

class IrisDataset(IrisDatasetBase):
    SUBSETS={
        'train_iris_nd_crosssensor_2013': {'size': 101968, 'num_classes': 940},
        'train_iris_casia_v4': {'size': 15890, 'num_classes': 1600},
        'train_iris_inno_a1': {'size': 79128, 'num_classes': 39659},
        'train_iris_nd_0405': {'size': 51880, 'num_classes': 570},
        'train_iris_utris_v1': {'size': 542, 'num_classes': 108},
    }
    CLASS_STRING_FORMAT='{__subset}:{subject_id}_{eye_side}'
    IMAGE_ID_STRING_FORMAT='{__subset}:{image_id}'
    def __init__(
        self, 
        root: str,
        subsets: list,
        pseudolabels: Optional[dict] = None,
        class_group_size: Optional[int] = 4,
        autocrop: Optional[bool] = True,
        unwrap: Optional[bool] = False,
        transform: Optional[Callable] = None,
    ) -> None:
        super(IrisDataset, self).__init__(
            root=root, autocrop=autocrop, 
            unwrap=unwrap, transform=transform
        )
        assert type(root) == str,\
            "Parameter root can must be string"
        assert all(type(s) == str for s in subsets),\
            "Subsets must be list of strings"
            
        self.subsets = subsets
        self.pseudolabels = pseudolabels
        self.class_group_size = class_group_size
        self.__load_subsets_anotations()
        self.__group_annotations_by_class()
        assert (
            self.expected_size == self.size and
            self.expected_num_classes == self.num_classes
        ), "Expected dataset sizes do not match"
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # #  OVERRIDDEN METHODS   # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    @property
    def size(self):
        return len(self.annotations)
    def get_img_annotation(self, index: int) -> dict:
        return self.annotations[index]
    def get_img_label(self, index: int) -> int:
        entry = self.get_img_annotation(index)
        if self.pseudolabels is None:
            label = entry['__class_number']
        else:
            label = self.pseudolabels[entry['__image_id']]
        return label
    def get_img_binary(self, index: int): 
        entry = self.get_img_annotation(index)
        img_path = os.path.join(
            self.root, entry['__subset'],
            entry['image_id'],'iris_right.UNKNOWN'
        )
        with open(img_path, "rb") as i:
            return i.read()
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # #  HELPER METHODS   # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def __load_subsets_anotations(self):
        self.annotations=[]
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
        entry_image_id_string = (
            IrisDataset.IMAGE_ID_STRING_FORMAT.format(**entry_dict)
        )
        if not entry_class_string in self.classes_dict:
            self.classes_dict[entry_class_string] = len(self.classes_dict)
        entry_dict['__image_id'] = entry_image_id_string
        entry_dict['__class_string'] = entry_class_string
        entry_dict['__class_number'] = self.classes_dict[entry_class_string]
        self.annotations.append(entry_dict)
    def __group_annotations_by_class(self):
        class_annotations_idx = {}
        for i, annotation in enumerate(self.annotations):
            if not annotation['__class_string'] in class_annotations_idx:
                class_annotations_idx[annotation['__class_string']] = []
            class_annotations_idx[annotation['__class_string']].append(i)
        argsort_idxs = []
        while len(class_annotations_idx) != 0:
            keys_list = list(class_annotations_idx.keys())
            random.shuffle(keys_list)
            for c in keys_list:
                l = class_annotations_idx[c]
                if len(l) <= self.class_group_size:
                    class_annotations_idx.pop(c)
                    a = l
                else:
                    a = l[:self.class_group_size]
                    class_annotations_idx[c] = l[self.class_group_size:]
                argsort_idxs.extend(a)
        annotations_tmp = [self.annotations[i] for i in argsort_idxs]
        self.annotations = annotations_tmp
        

    @property
    def num_classes(self):
        return len(self.classes_dict)
    @property
    def expected_size(self):
        return sum(IrisDataset.SUBSETS[s]['size'] for s in self.subsets)
    @property
    def expected_num_classes(self):
        return sum(IrisDataset.SUBSETS[s]['num_classes'] for s in self.subsets)


class IrisVerificationDatasetPseudo(IrisDataset):
    def __init__(
        self, 
        root: str,
        subsets: list,
        pseudolabels: Optional[dict] = None,
        autocrop: Optional[bool] = True,
        unwrap: Optional[bool] = False,
        transform: Optional[Callable] = None,
    ) -> None:
        super(IrisVerificationDatasetPseudo, self).__init__(
            root=root, subsets=subsets, pseudolabels=pseudolabels, 
            autocrop=autocrop, unwrap=unwrap, transform=transform,
        )
    def get_img_label(self, index: int) -> str:
        return self.get_img_annotation(index)['__image_id']

class DatasetSubset(VisionDataset):
    def __init__(
        self, 
        dataset: VisionDataset,
        indicies: list,
        transform: Optional[Callable]=None,
    ) -> None:
        super(DatasetSubset, self).__init__(dataset.root, transform=transform)
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