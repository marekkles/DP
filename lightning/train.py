from copyreg import pickle
from distutils.log import warn
import os
import pickle
import pytorch_lightning as pl
import torch
from torchvision import transforms
from dataset import *
from evaluation import pairs_impostor_scores
import models
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CSVLogger

def name_gen(timestamp: int):
    available_chars=[
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 
        'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', 
        '5', '6', '7', '8', '9', '0'
    ]
    ts = timestamp
    name = []
    while ts != 0:
        idx = ts % len(available_chars)
        ts = ts // len(available_chars)
        name.append(available_chars[idx])
    return ''.join(name)
        
def main(args, mode):
    
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
    loggers = [
        CSVLogger(save_dir=args["run_root_dir"], 
                  name=None, version='csvs'),
    ]
    if args['use_wandb']:
        loggers.append(
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
            save_top_k=1,
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
        num_workers=args["num_workers"],
        auto_crop=args["auto_crop"],
        unwrap=args["unwrap"],
        shuffle=args["shuffle"],
    )
    # model
    model = models.__dict__[args["model"]](**args["model_args"])
    if "resume_checkpoint" in args:
        model.load_from_checkpoint(args["resume_checkpoint"])

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
        callbacks=callbacks,
        gradient_clip_val=5, gradient_clip_algorithm="norm"
    )
    if mode == "train" or mode == "train+evaluate+export":
        trainer.fit(
            model,
            data_loader
        )
    if mode == "export" or mode == "train+evaluate+export":
        encoder_state_dict = model.encoder.state_dict()
        torch.save(encoder_state_dict, os.path.join(
            args["run_root_dir"],
            'encoder-{}.pickle'.format(args["run_name"])
        ))
    if mode == "evaluate" or mode == "train+evaluate+export":
        data = trainer.predict(
            model,
            data_loader,
            return_predictions=True
        )
        vectors = {}
        for vec,val in data:
            vectors.update(zip(
                [k for k in val], 
                [l.cpu().detach().numpy() for l in vec]
            ))
        
        with open(os.path.join(
            args["run_root_dir"],
            'vectors-{}.pickle'.format(args["run_name"])
        ), "wb") as f:
            pickle.dump(vectors, f)
    
    print("Finished running")

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
        choices=["train", "evaluate", "export", "train+evaluate+export"],
        type=str,
        default="train+evaluate+export",
        help=f"Mode to train or evaluate (default: train+evaluate+export)"
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
        
        args_file["run_name"] = '{}-{}'.format(
            args_file['model'],
            args_file['model_args']['backbone'])
        if 'metric' in args_file['model_args']:
            args_file["run_name"] += '-{}'.format(args_file['model_args']['metric'])
        args_file["run_name"] += '-{}'.format(name_gen(time.time_ns()//1_000_000_000))

        args_file["run_root_dir"] = os.path.join(
            args_file["root_dir"], args_file["run_name"]
        )

        if not os.path.exists(args_file["run_root_dir"]):
            os.makedirs(args_file["run_root_dir"])

        with open(os.path.join(args_file["run_root_dir"],"args.yaml"), 'w') as f:
            yaml.dump(args_file, f)

    main(args=args_file, mode=args_program.mode)