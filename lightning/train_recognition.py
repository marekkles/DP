
from asyncio import protocols
from asyncio.log import logger
from os import access
import pytorch_lightning as pl
import torch
from torchvision import transforms
from dataset import *
from recognition_model import RecognitionNet
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

args = {
    "experiment_name":"iresnet50-ArcFace",
    "project_name":"dp",

    "backbone":"iresnet50",
    "backbone_args":{
        "in_channels" : 1, 
        "num_classes" : 512, 
        "dropout_prob0" : 0.5
    }, 
    "metric":"ArcFaceDecoder", 
    "metric_args":{
        "in_features" : 512, 
        "out_features" : 3218
    },
    "optim":"SGD",
    "optim_args":{
        "lr":1e-3
    },
    "lr_scheduler":"StepLR",
    "lr_scheduler_args":{
        "step_size":30, 
        "gamma":0.1
    },
    "max_epochs":30,
    "batch_size":32,
    "num_workers":4,
    "accelerator":"cpu",
    "devices":1,
    "datasets" : [
        "../Datasets/train_iris_nd_crosssensor_2013", 
        "../Datasets/train_iris_casia_v4",
        "../Datasets/train_iris_nd_0405",
        "../Datasets/train_iris_utris_v1"
    ],
    "predic_dataset" : "../Datasets/iris_verification_NDCSI2013_01_05",
    "train_transform" : {
        "Resize" : {"size":[112,112]},
        "RandomInvert" : {"p":0.2},
        "Normalize" : {"mean":[0.485], "std":[0.229]},
        "RandomAdjustSharpness"  : {"sharpness_factor":3,"p":0.5},
        "RandomAutocontrast" : {"p":0.5},
        "RandomAffine": {
            "degrees":8, 
            "translate":[0.1, 0.1],
            "scale":[0.8, 1.33],
            "shear":10,
            "fill":0
        },
        "RandomErasing":{
            "p":0.5, 
            "scale":[0.02, 0.33],
            "ratio":[0.3, 3.3],
            "value":0, 
            "inplace":False
        }
    },
    "val_transform" : {
        "Resize" : {"size":[112,112]},
        "Normalize" : {"mean":[0.485], "std":[0.229]},
    },
    "test_transform" : {
        "Resize" : {"size":[112,112]},
        "Normalize" : {"mean":[0.485], "std":[0.229]},
    },
    "predict_transform" : {
        "Resize" : {"size":[112,112]},
        "Normalize" : {"mean":[0.485], "std":[0.229]},
    }
}

def main(args):
    #transforms
    #train transforms
    train_transform = transforms.Compose([
        transforms.Resize(**args["train_transform"]["Resize"]),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomInvert(**args["train_transform"]["RandomInvert"]),
        transforms.Normalize(**args["train_transform"]["Normalize"]),
        transforms.RandomAdjustSharpness(
            **args["train_transform"]["RandomAdjustSharpness"]
        ),
        transforms.RandomAutocontrast(
            **args["train_transform"]["RandomAutocontrast"]
        ),
        transforms.RandomAffine(**args["train_transform"]["RandomAffine"]),
        transforms.RandomErasing(**args["train_transform"]["RandomErasing"]),
    ])
    #val transforms
    val_transform=transforms.Compose([
        transforms.Resize(**args["val_transform"]["Resize"]),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(**args["val_transform"]["Normalize"]),
    ])
    #test transforms
    test_transform=transforms.Compose([
        transforms.Resize(**args["test_transform"]["Resize"]),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(**args["test_transform"]["Normalize"]),
    ])
    #predict transforms 
    predict_transform=transforms.Compose([
        transforms.Resize(**args["predict_transform"]["Resize"]),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(**args["predict_transform"]["Normalize"]),
    ])
    #logger
    logger = WandbLogger(name='iresnet50',project='dp')
    #logger = TensorBoardLogger('tensorboard')
    #callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath = "model_checkpoints",
        ),
    ]
    # data
    data_loader = IrisDataModule(
        data_dir= args["datasets"],
        predic_data_dir=args["predic_dataset"],
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        predict_transform=predict_transform,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"]
    )
    # model
    model = RecognitionNet(
        backbone=args["backbone"], 
        backbone_args=args["backbone_args"], 
        metric=args["metric"], 
        metric_args=args["metric_args"],
        optim=args["optim"], 
        optim_args=args["optim_args"],
        lr_scheduler=args["lr_scheduler"],
        lr_scheduler_args=args["lr_scheduler_args"],
    )
    # training
    trainer = pl.Trainer(
        max_epochs=args["max_epochs"],
        accelerator=args["accelerator"],
        devices=args["devices"], 
        logger=logger, 
        callbacks=callbacks
    )
    trainer.fit(model, data_loader)


if __name__ == "__main__":
    main(args=args)