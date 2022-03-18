import os
import csv
import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional, Tuple, Any
import io
import math

import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from PIL import Image

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
    def __get_img(self, index: int) -> Tuple[torch.Tensor]:
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
        img = self.__get_img(index)
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
        autocrop: bool = True,
        unwrap: bool = False,
        transform: Optional[Callable] = None
    ) -> None:
        super(IrisVerificationDataset, self).__init__(
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
    def __init__(
        self, 
        root: str,
        subsets: list,
        autocrop: bool = True,
        unwrap: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        super(IrisDataset, self).__init__(
            root, autocrop, unwrap, transform
        )
        assert type(root) == str,\
            "Parameter root can must be string"
        assert all(type(s) == str for s in subsets),\
            "Subsets must be list of strings"
            
        self.subsets = subsets
        self.__load_subsets_anotations()
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
    def img_annotation(self, index: int) -> dict:
        return self.annotations[index]
    def img_label(self, index: int) -> int:
        entry = self.get_img_annotation(index)
        return entry['__class_number']
    def img_binary(self, index: int): 
        entry = self.entry_list[index]
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
        if not entry_class_string in self.classes_dict:
            self.classes_dict[entry_class_string] = len(self.classes_dict)
        entry_dict['__class_string'] = entry_class_string
        entry_dict['__class_number'] = self.classes_dict[entry_class_string]
        self.annotations.append(entry_dict)
    @property
    def num_classes(self):
        return len(self.classes_dict)
    @property
    def expected_size(self):
        return sum(IrisDataset.SUBSETS[s]['size'] for s in self.subsets)
    @property
    def expected_num_classes(self):
        return sum(IrisDataset.SUBSETS[s]['num_classes'] for s in self.subsets)
    

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