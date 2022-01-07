import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
from backbones import *
from datasets import *
from arcface.losses import *
import utils
import presets
import time
import datetime

class ArcFaceMetric(torch.nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceMetric, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = ArcFace(s, m)
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, input, label):
         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
         return self.fc(cosine, label)

def train_one_epoch(model: torch.nn.Module, metric_fc: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, data_loader: DataLoader, device: torch.device, epoch: int, args):
    model.train()
    metric_fc.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    header = f"Epoch: [{epoch}]"

    print(f"Epoch: [{epoch}]")
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        feature = model(image)
        output = metric_fc(feature, target)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = image.shape[0]
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

def evaluate(model: torch.nn.Module, metric_fc: torch.nn.Module, criterion: torch.nn.Module, data_loader: DataLoader, device: torch.device, print_freq=100):
    model.eval()
    metric_fc.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    header = f"Test: "

    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            start_time = time.time()
            image, target = image.to(device), target.to(device)
            feature = model(image)
            output = metric_fc(feature, target)
            loss = criterion(output, target)
            batch_size = image.shape[0]
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    
    metric_logger.synchronize_between_processes()
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")

def load_data(dir, args):
    print("Loading data")
    
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    
    interpolation = InterpolationMode(args.interpolation)
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)

    dataset_train = IrisDataset([dir], transform=presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
            ),
            image_set="train"
        )
    
    print("Loading data")
    dataset_val = IrisDataset([dir], transform=presets.ClassificationPresetEval(
                crop_size=val_crop_size, 
                resize_size=val_resize_size, 
                interpolation=interpolation
            ), 
            image_set="val"
        )
    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=None,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=args.batch_size, 
        sampler=test_sampler, 
        num_workers=args.workers, 
        pin_memory=True
    )
    return dataset_train, dataset_val, train_sampler, test_sampler, data_loader_train, data_loader_val

def main(args):
    print(args)

    device = torch.device(args.device)

    dataset, dataset_test, train_sampler, test_sampler, data_loader, data_loader_test = load_data(args.data_path, args)
    
    print("Creating model")
    
    model = iresnet18(num_features=args.embedding_size, dropout=0.5)
    model.to(device)
    metric_fc = ArcFaceMetric(args.embedding_size, dataset.num_classes)
    metric_fc.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    print("Creating optimizer")
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, metric_fc, criterion, optimizer, data_loader, device, epoch, args)
        lr_scheduler.step()
        evaluate(model, metric_fc, criterion, data_loader_test, device=device)
        #save checkpoint
        if args.output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "metric_fc":metric_fc.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training, using ArcFace", add_help=add_help)

    #parser.add_argument("--data-path", default="../Datasets/train_iris_nd_crosssensor_2013", type=str, help="dataset path")
    parser.add_argument("--data-path", default="../Datasets/train_iris_casia_v4", type=str, help="dataset path")

    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--embedding-size", default=128, type=int, help="size of emdedding space (default: 128)")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu Default: cpu)")
    parser.add_argument(
        "-b", "--batch-size", default=64, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=112, type=int, help="the resize size used for validation (default: 128)"
    )
    parser.add_argument(
        "--val-crop-size", default=112, type=int, help="the central crop size used for validation (default: 90)"
    )
    parser.add_argument(
        "--train-crop-size", default=112, type=int, help="the random crop size used for training (default: 90)"
    )
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)