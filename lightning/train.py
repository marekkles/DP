from copyreg import pickle
from distutils.log import warn
import os
import pickle
import numpy as np
import pytorch_lightning as pl
import torch
from torchvision import transforms
from dataset import *
import models
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from utils import *

def main(args, mode):    
    loggers = [
        CSVLogger(save_dir=args["run_root_dir"], 
                  name=None, version='csvs'),
    ]
    
    if args['use_wandb'] and (mode == "train" or mode == "train+evaluate+export"):
        print(f"Using Weights&Biases under project {args['project_name']} ")
        loggers.append(
            WandbLogger(
                name=args["run_name"],
                save_dir=args["run_root_dir"],
                project=args["project_name"]
            )
        )
        
    callbacks = [
        pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=os.path.join(args["run_root_dir"], "checkpoints"),
            monitor='val_loss',
            save_last=True,
            save_top_k=1,
            mode='min',
            auto_insert_metric_name=True,
            filename='checkpoint-{epoch}-{val_loss:.2f}'
        )
    ]
    # data
    data_loader = IrisDataModule(
        root_dir=args["dataset_root"],
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        num_in_channels=args["num_in_channels"],
        train_subset=args["dataset_subsets"],
        train_pseudolabels=args["train_pseudolabels"] if "train_pseudolabels" in args else None,
        auto_crop=args["auto_crop"],
        unwrap=args["unwrap"],
        shuffle=args["shuffle"],
        train_transform=train_transform(**args["train_transform"]),
        val_transform=val_transform(**args["val_transform"]),
        test_transform=test_transform(**args["test_transform"]),
        predict_transform=predict_transform(**args["predict_transform"]),
    )
    
    if "resume_checkpoint" in args:
        print(f"Resuming from checkpoint {args['resume_checkpoint']} ")
        model = models.__dict__[args["model"]].load_from_checkpoint(
            args["resume_checkpoint"], hparams_file=args_file["resume_hparams"]
        )
    else:
        print(f"Creating new model")
        model = models.__dict__[args["model"]](**args["model_args"])

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
        gradient_clip_val=5, gradient_clip_algorithm="norm",
    )
    if mode == "train" or mode == "train+evaluate+export":
        print(f"Starting trainning")
        trainer.fit(
            model,
            data_loader,
            ckpt_path=args["resume_checkpoint"] if "resume_checkpoint" in args else None
        )
    if mode == "export" or mode == "train+evaluate+export":
        print("Exporting checkpoints backbone")
        encoder_state_dict = model.encoder.state_dict()
        torch.save(encoder_state_dict, os.path.join(
            args["run_root_dir"],
            'encoder-{}.pickle'.format(args["run_name"])
        ))
    if mode == "evaluate" or mode == "train+evaluate+export":
        print("Evaluating model")
        for predict_dataset in data_loader.iris_predict_list:
            print(f"Predict for dataset: {predict_dataset}")
            data_loader.iris_predict_select(predict_dataset)
            data = trainer.predict(
                model,
                data_loader,
                return_predictions=True
            )
            #unpack results
            results = unpack_results(data)
            #If evaluated output has label field
            if "label" in results:
                save_labeled_results(
                    results, args["run_root_dir"], 
                    args["run_name"], predict_dataset
                )            
            #If evaluated output is unlabeled
            else:
                save_unlabeled_results(
                    results, args["run_root_dir"], 
                    args["run_name"], predict_dataset
                )
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
        args_file["resume_hparams"] = os.path.join(
            args_file['resume_dir'], "csvs",
            "hparams.yaml"
        )
        if args_file["run_root_dir"] != args_file['resume_dir']:
            warn('Resume directory is different as in argsfile')
            args_file["run_root_dir"] = args_program.resume_dir
    else:
        with open(args_program.args_path, 'r') as f:
            args_file = yaml.load(f, yaml.FullLoader)
        
        args_file["run_name"] = args_file['model']
        if 'backbone' in args_file['model_args']:
            args_file["run_name"] += '-' + args_file['model_args']['backbone']
        if 'metric' in args_file['model_args']:
            args_file["run_name"] += '-' + args_file['model_args']['metric']
        args_file["run_name"] += '-' + name_gen(time.time_ns()//1_000_000_000)

        args_file["run_root_dir"] = os.path.join(
            args_file["root_dir"], args_file["run_name"]
        )

        if not os.path.exists(args_file["run_root_dir"]):
            os.makedirs(args_file["run_root_dir"])

        with open(os.path.join(args_file["run_root_dir"],"args.yaml"), 'w') as f:
            yaml.dump(args_file, f)

    main(args=args_file, mode=args_program.mode)