from copyreg import pickle
from distutils.log import warn
from ntpath import join
import os
import joblib
import pytorch_lightning as pl
import torch
from torchvision import transforms
from dataset import *
from recognition_model import RecognitionNet
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

def main(args, mode: str):
    #transforms
    #train transforms
    train_transform=transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomAffine(**args["train_transform"]["RandomAffine"]),
        transforms.Resize(**args["train_transform"]["Resize"]),
        transforms.RandomAdjustSharpness(
            **args["train_transform"]["RandomAdjustSharpness"]
        ),
        transforms.RandomAutocontrast(
            **args["train_transform"]["RandomAutocontrast"]
        ),
        transforms.RandomInvert(**args["train_transform"]["RandomInvert"]),
        transforms.Normalize(**args["train_transform"]["Normalize"]),
        transforms.RandomErasing(**args["train_transform"]["RandomErasing"]),
    ])
    #val transforms
    val_transform=transforms.Compose([
        transforms.Resize(**args["val_transform"]["Resize"]),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(**args["val_transform"]["Normalize"]),
    ])
    #test transforms
    test_transform=transforms.Compose([
        transforms.Resize(**args["test_transform"]["Resize"]),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(**args["test_transform"]["Normalize"]),
    ])
    #predict transforms 
    predict_transform=transforms.Compose([
        transforms.Resize(**args["predict_transform"]["Resize"]),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(**args["predict_transform"]["Normalize"]),
    ])
    #logger
    loggers = (
        WandbLogger(
            name=args["run_name"],
            save_dir=args["run_root_dir"],
            project=args["project_name"]
        )
    )
    #callbacks
    callbacks = [
        pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=os.path.join(args["run_root_dir"], "checkpoints"),
            monitor='val_loss',
            save_last=True,
            save_top_k=5,
            mode='min',
            auto_insert_metric_name=True,
            filename='checkpoint-{epoch}-{val_loss:.2f}'
        ),
        #pl.callbacks.EarlyStopping(monitor="val_loss"),
        #pl.callbacks.DeviceStatsMonitor(),
    ]
    # data
    data_loader = IrisDataModule(
        data_dir=args["dataset_root"],
        subsets=args["dataset_subsets"],
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
        default_root_dir=args["run_root_dir"],
        log_every_n_steps=args["log_steps"],
        accumulate_grad_batches=args["grad_batches"],
        logger=loggers,
        callbacks=callbacks
    )
    if mode == "train":
        trainer.fit(
            model,
            data_loader,
            ckpt_path=(
                args["resume_checkpoint"] if "resume_checkpoint" in args else None
            )
        )
    elif mode == "evaluate":
        data = trainer.predict(
            model,
            data_loader,
            ckpt_path=(
                args["resume_checkpoint"] if "resume_checkpoint" in args else None
            ),
            return_predictions=True
        )
        res = {}
        for vec,val in data:
            res.update(zip(
                val.cpu().detach().numpy().tolist(), 
                [l.cpu().detach().numpy() for l in vec]
            ))
        import pickle
        with open(os.path.join(
            args["resume_dir"],  
            'prediction-{}.pickle'.format(args["run_name"])
        ), "wb") as f:
            pickle.dump(res, f)
    else:
        assert False, "Not implemented!"

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description="Recognition Training script", 
        add_help=add_help
    )
    parser.add_argument(
        '--args-path',
        type=str,
        default='args.yaml',
        help=f"Path to argument file for training (default: args.yaml)"
    )
    parser.add_argument(
        '--mode',
        choices=["train", "evaluate"],
        type=str,
        default="train",
        help=f"Mode to train or evaluate (default: train)"
    )
    parser.add_argument(
        '--resume-dir',
        type=str,
        help=f"Path to resume directory"
    )
    parser.add_argument(
        '--resume-checkpoint',
        default='last.ckpt',
        type=str,
        help=f"Resume checkpoint (default: last.ckpt)"
    )
    return parser


if __name__ == "__main__":
    args_program = get_args_parser().parse_args()
    import yaml
    import time

    if args_program.resume_dir != None:
        args_program.args_path = os.path.join(
            args_program.resume_dir, 'args.yaml'
        )

        with open(args_program.args_path, 'r') as f:
            args_file = yaml.load(f,yaml.FullLoader)
        
        args_file['resume_dir'] = args_program.resume_dir
        args_file['resume_checkpoint'] = os.path.join(
            args_file['resume_dir'], "checkpoints",
            args_program.resume_checkpoint
        )
        if args_file["run_root_dir"] != args_file['resume_dir']:
            warn('Resume directory is different as in argsfile')
            args_file["run_root_dir"] = args_program.resume_dir
    else:
        with open(args_program.args_path, 'r') as f:
            args_file = yaml.load(f,yaml.FullLoader)
        
        args_file["run_name"] = '{}-{}-{}'.format(
            args_file['backbone'],
            args_file['metric'],
            time.time_ns()//1_000_000_000
        )

        args_file["run_root_dir"] = os.path.join(
            args_file["root_dir"], args_file["run_name"]
        )

        if not os.path.exists(args_file["run_root_dir"]):
            os.makedirs(args_file["run_root_dir"])

        with open(os.path.join(args_file["run_root_dir"],"args.yaml"), 'w') as f:
            yaml.dump(args_file, f)

    main(args=args_file, mode=args_program.mode)