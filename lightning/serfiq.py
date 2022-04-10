import torch
import os

from torch._C import dtype
from backbones import *
from dataset import IrisVerificationDataset
from torchvision.transforms import transforms
import time
import datetime
import serfiq
import dataset
import utils

def main(model, datamodule):
    for predict_ds in datamodule.iris_predict_list:
        print(f"Predict for dataset: {predict_dataset}")
        datamodule.iris_predict_select(predict_ds)
        data = 
    pass


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="ArcFace model evaluation :get: embeddings", add_help=add_help)
    parser.add_argument("--run-root", default=os.path.join("runs", "RecognitionNet-magiresnet50-ArcFaceLoss-arzKXb"), type=str, help="Run root")
    parser.add_argument("--device", default="cpu", type=str, help="Device to use")
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    model, args_file = serfiq.load_model(args.run_root)

    datamodule = dataset.IrisDataModule(args_file['dataset_root'], 1, 1, 1,
        train_pseudolabels= None, 
        train_subset= args_file['dataset_subsets'], 
        auto_crop= args_file['autocrop'], 
        unwrap= args_file['unwrap'], 
        shuffle= args_file['shuffle'], 
        train_transform= dataset.train_transform(**args_file['train_transform']), 
        val_transform= dataset.train_transform(**args_file['val_transform']),
        test_transform= dataset.train_transform(**args_file['test_transform']), 
        predict_transform= dataset.train_transform(**args_file['predict_transform'])
    )
    serfiq.setup_model(model)
    device = torch.device(args.device)
    model = model.to(device)
    main(model, datamodule)
    

