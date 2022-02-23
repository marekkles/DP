
from asyncio import protocols
from asyncio.log import logger
from os import access
import pytorch_lightning as pl
import torch
from torchvision import transforms
from dataset import *
from recognition_model import RecognitionNet
from pytorch_lightning.loggers import WandbLogger

#logger
wandb_logger = WandbLogger(name='iresnet-50',project='dp')
# data
data_loader = IrisDataModule(
    data_dir= [
        "../Datasets/train_iris_nd_crosssensor_2013", 
        "../Datasets/train_iris_casia_v4",
        "../Datasets/train_iris_nd_0405",
        "../Datasets/train_iris_utris_v1"
    ],
    predic_data_dir="../Datasets/iris_verification_NDCSI2013_01_05",
    train_transforms=transforms.Compose([
        transforms.Resize((112,112)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomInvert(p=0.5),
        transforms.Normalize(mean=[0.485], std=[0.229]),
        transforms.RandomAdjustSharpness(sharpness_factor=2,p=0.5),
        transforms.RandomAutocontrast(p=0.5),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.8, 1.3),
            shear=20,
            fill=0,
        ),
        transforms.RandomErasing(
            p=0.5, 
            scale=(0.02, 0.33), 
            ratio=(0.3, 3.3), 
            value=0, 
            inplace=False
        ),
    ]), 
    predict_transforms=transforms.Compose([
        transforms.Resize((112,112)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
)
# model
model = RecognitionNet(
    backbone="iresnet50", 
    backbone_args={"in_channels" : 1, "num_classes" : 512, "dropout_prob0" : 0.5}, 
    metric="ArcFaceDecoder", 
    metric_args={"in_features" : 512, "out_features" : data_loader.num_classes}
)
# training
trainer = pl.Trainer(max_epochs=5, logger=wandb_logger)
trainer.fit(model, data_loader)
