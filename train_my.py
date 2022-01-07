import torch
import torch.nn as nn
import torch.functional as F
from backbones import *
from datasets import *
from arcface.losses import *
from configs import *

def main(config):
    device = torch.device("cpu")
    train_dataset = IrisDataset("../Datasets/train_iris_nd_crosssensor_2013", image_set='train')
    val_dataset = IrisDataset("../Datasets/train_iris_nd_crosssensor_2013", image_set='val')

    pass


if __name__ == '__main__':
    main(base_config())