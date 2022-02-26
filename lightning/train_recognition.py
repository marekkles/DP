import os
import pytorch_lightning as pl
import torch
from torchvision import transforms
from dataset import *
from recognition_model import RecognitionNet
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

def main(args):
    #log args
    if not os.path.exists(args["root_dir"]):
        os.makedirs(args["root_dir"])
    with open(os.path.join(args["root_dir"],"args.yaml"), 'w') as f:
        import yaml
        yaml.dump(args, f)
    #transforms
    #train transforms
    train_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomAffine(**args["train_transform"]["RandomAffine"]),
        transforms.RandomAdjustSharpness(
            **args["train_transform"]["RandomAdjustSharpness"]
        ),
        transforms.RandomAutocontrast(
            **args["train_transform"]["RandomAutocontrast"]
        ),
        transforms.RandomErasing(**args["train_transform"]["RandomErasing"]),
        transforms.Resize(**args["train_transform"]["Resize"]),
        #transforms.RandomInvert(**args["train_transform"]["RandomInvert"]),
        transforms.Normalize(**args["train_transform"]["Normalize"]),
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
    logger = WandbLogger(
        name=args["run_name"],
        save_dir=args["root_dir"],
        project=args["project_name"]
    )
    #logger = TensorBoardLogger('tensorboard')
    #callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath = os.path.join(args["root_dir"], "model_checkpoints"),
        ),
        pl.callbacks.EarlyStopping(monitor="val_loss"),
        pl.callbacks.DeviceStatsMonitor(),

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
        num_nodes=args["num_nodes"],
        default_root_dir=args["root_dir"],
        logger=logger,
        callbacks=callbacks
    )
    trainer.fit(model, data_loader)

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description="Recognition Training script", 
        add_help=add_help
    )
    args_path_default=os.path.join(os.path.dirname(__file__),'args.yaml')
    parser.add_argument(
        '--args-path', 
        default=args_path_default,
        type=str,  
        help=f"datasets paths (default: {args_path_default})"
    )
    return parser


if __name__ == "__main__":
    args_program = get_args_parser().parse_args()
    #print(args)
    with open(args_program.args_path, 'r') as f:
        import yaml
        args_file = yaml.load(f,yaml.FullLoader)
    main(args=args_file)