from arcface.backbones import *
import datetime
import os
import time
import warnings

import torch
import torch.utils.data
import torchvision
from torchvision.transforms import autoaugment, transforms
from torch import nn

from datasets import IrisVerificationDataset
from backbones import get_model

def main(args):
    print(f"Loading model from {args.model_path}",end="")
    path = os.path.join(args.model_path)
    
    device = torch.device(args.device)
    checkpoint = torch.load(path, map_location=device)
    
    if args.model == "iresnet18":
        model = iresnet18(num_features=args.embedding_size, dropout=0.5)
    elif args.model == "iresnet50":
        model = iresnet50(num_features=args.embedding_size, dropout=0.5)
    else:
        assert False, f"Cannot create given model {args.model}"

    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    print(": done")

    print("Loading dataset", end="")
    dataset_eval = IrisVerificationDataset(args.data_path, transform=transforms.Compose([
        transforms.Resize(112),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]))
    eval_sampler = torch.utils.data.SequentialSampler(dataset_eval)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_eval, 
        batch_size=args.batch_size, 
        sampler=eval_sampler, 
        num_workers=args.workers, 
        pin_memory=True
    )
    print(": done")

    
    with torch.no_grad():
        print("Creating embeddings")
        start = time.time()
        embedding_dict = {}
        for bn, (image, target) in enumerate(data_loader_val):
            time_passed = datetime.timedelta(seconds=int(time.time() - start))
            time_est = None if bn == 0 else time_passed/(bn*args.batch_size) * len(dataset_eval)
            print(f"Processed: {bn*args.batch_size/len(dataset_eval)*100:.2f}%, est: {time_est} [h:m:s]\r", end="")
            image = image.to(device)
            embedding = model(image)
            for i in range(embedding.shape[0]):
                embedding_dict[target[i].item()] = embedding[i].detach()
    
        print("Evaluating impostors distances")
        impostor_dict = {"euclidean":{},"cosine":{}}
        for i, (a, b) in enumerate(dataset_eval.impostors):
            print(f"Processed: {i/len(dataset_eval.impostors)*100:.2f}%\r", end="")
            if (not a in embedding_dict) or (not b in embedding_dict):
                continue
            impostor_dict["euclidean"][(a,b)] = (embedding_dict[a] - embedding_dict[b]).pow(2).sum().pow(0.5).item()
            impostor_dict["cosine"][(a,b)] = torch.nn.functional.cosine_similarity(embedding_dict[a], embedding_dict[b]).item()
    
        print("Evaluating pairs distances")
        pair_dict = {"euclidean":{},"cosine":{}}
        for i, (a, b) in enumerate(dataset_eval.pairs):
            print(f"Processed: {i/len(dataset_eval.pairs)*100:.2f}%\r", end="")
            if (not a in embedding_dict) or (not b in embedding_dict):
                continue
            pair_dict["euclidean"][(a,b)] = (embedding_dict[a] - embedding_dict[b]).pow(2).sum().pow(0.5).item()
            pair_dict["cosine"][(a,b)] = torch.nn.functional.cosine_similarity(embedding_dict[a], embedding_dict[b]).item()

    
    print(f"\nSaving results to {args.output}")
    savepoint = {"embeddings":embedding_dict, "impostor_scores": impostor_dict, "pair_scores": pair_dict}
    print(f"Saved:\n\t{len(embedding_dict)} embeddings\n\t{len(impostor_dict)} impostors scores\n\t{len(pair_dict)} pairs scores")
    torch.save(savepoint, args.output)

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="ArcFace model evaluation :get: embeddings", add_help=add_help)

    parser.add_argument("--data-path", default="../Datasets/iris_verification_NDCSI2013_01_05", type=str, help="dataset path")
    parser.add_argument("--model", default="iresnet18", type=str, help="model name")
    parser.add_argument("--model-path", default=os.path.join("..","models","iresnet18","model_47.pth"), type=str, help="path to model .pth file")
    parser.add_argument("--embedding-size", default=128, type=int, help="size of emdedding space (default: 128)")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu Default: cpu)")
    parser.add_argument(
        "-b", "--batch-size", default=128, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument(
        "-o", "--output", default="embeddings.pth", type=str, help="output emdeddings file (default: embeddings.pth)"
    )

    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
    
