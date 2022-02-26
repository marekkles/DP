from logging import root
from cv2 import transform
from torchvision.datasets.vision import VisionDataset
import torch
from collections.abc import Callable
from typing import Optional, Tuple, Any
import os
import csv
from .utils import IrisImage


from torchvision.datasets.vision import VisionDataset
from collections.abc import Callable
from typing import Optional, Tuple, Any

class IrisVerificationDataset(VisionDataset):
    def __init__(
        self, 
        root: list,
        autocrop: bool = True,
        transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transform=transform)
        
        self.anotations = None
        self.autocrop = autocrop
        self.tmp_img = IrisImage()
        self.mapping    = None
        self.pairs      = None
        self.impostors  = None

        with open(os.path.join(root, 'annotations.csv'), newline='') as csvfile:
            csv_file = csv.reader(csvfile, delimiter=',', quotechar='"')
            self.anotations = list(csv_file)
        with open(os.path.join(root, 'mapping.csv'), newline='') as csvfile:
            csv_file = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(csv_file)
            self.mapping = list(csv_file)
        with open(os.path.join(root, 'pairs.csv'), newline='') as csvfile:
            csv_file = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(csv_file)
            self.pairs = list(map(lambda x: (int(x[0]), int(x[1])), csv_file ))
        with open(os.path.join(root, 'impostors.csv'), newline='') as csvfile:
            csv_file = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(csv_file)
            self.impostors = list(map(lambda x: (int(x[0]), int(x[1])), csv_file ))
        
        self.mapping = dict(map(lambda x: (int(x[1]),x[0]), self.mapping))

        self.header = self.anotations[0]
        self.image_id_idx = self.header.index('image_id')

        self.anotations_map = dict(map(lambda i: (i[self.image_id_idx], i), self.anotations[1:]))

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types
        """
        anotation_index = index+1
        param_list = self.anotations_map[self.mapping[anotation_index]]
        param_list[self.image_id_idx] = str(anotation_index)
        dataset_root = self.root

        self.tmp_img.update(self.header,param_list,dataset_root)
        
        if self.autocrop:
            self.tmp_img.crop()

        img = self.tmp_img.get()

        if self.transform is not None:
            img = self.transform(img)
        
        return img, anotation_index
    def __len__(self) -> int:
        return len(self.mapping)

class IrisDataset(VisionDataset):
    def __init__(
        self, 
        root: list,
        mode: str = "classification",
        autocrop: bool = True,
        transform: Optional[Callable] = None, 
        transforms: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transform=transform, transforms=transforms, target_transform=target_transform)
        
        modes = {
            "classification",
            "segmentation"
        }

        assert mode in modes, f"Unsupported dataset mode {mode}, allowed option are {', '.join(i for i in modes)}"
        assert type(root) == list and all(type(s) == str for s in root), "Parameter root can must be list of strings"

        self.mode = mode
        self.autocrop = autocrop
        self.tmp_img = IrisImage()
        self.anotations = dict()
        self.subjects_set = dict()
        self.subjects = list()
        
        self.picked_subjects = None

        self.pairs = list()
        self.impostors = list()

        for dataset_dir in root:
            with open(os.path.join(dataset_dir, 'annotations.csv'), newline='') as csvfile:
                annotations = csv.reader(csvfile, delimiter=',', quotechar='"')
                self.anotations[dataset_dir] = list(annotations)
            
            header = self.anotations[dataset_dir][0]
            subject_id_idx = header.index('subject_id')
            eye_side_idx = header.index('eye_side')

            for i, entry in enumerate(self.anotations[dataset_dir][1:]):
                subject_id = entry[subject_id_idx]
                eye_side = entry[eye_side_idx]
                subject_id_combined = f"{subject_id}_{eye_side}"

                if not subject_id_combined in self.subjects_set:
                    self.subjects_set[subject_id_combined] = list()
                
                self.subjects_set[subject_id_combined].append(len(self.subjects))
                self.subjects.append( (subject_id_combined, dataset_dir, i+1) )

        self.subject_to_id_dict = dict(map((lambda x : (x[1],x[0])), enumerate(self.subjects_set)))
        self.id_to_subject_dict = dict(enumerate(self.subjects_set))
        
        self.picked_subjects = self.subjects
        self.num_classes = len(self.subjects_set)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types
        """

        subject = self.picked_subjects[index]
        header = self.anotations[subject[1]][0]
        param_list = self.anotations[subject[1]][subject[2]]
        dataset_root = subject[1]

        self.tmp_img.update(header,param_list,dataset_root)
        
        if self.autocrop:
            self.tmp_img.crop()

        img = self.tmp_img.get()

        if self.transform is not None:
            img = self.transform(img)

        if self.mode == "classification":
            target = self.subject_to_id_dict[subject[0]]
            if self.target_transform is not None:
                target = self.target_transform(target)
        elif self.mode == "segmentation":
            self.tmp_img.update_mask()
            target = self.tmp_img.get_mask()
            if self.transforms is not None:
                img, target = self.transforms(img, target)
        else:
            assert False, "Unsupported dataset mode"

        return img, target
    def __len__(self) -> int:
        return len(self.picked_subjects)



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
        if transform != None:
           img = self.transform(img)
        return img, target
    def __len__(self) -> int:
        return len(self.indicies)