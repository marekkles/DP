import torch
import os

from torch._C import dtype
from backbones import *
from datasets import *
from torchvision.transforms import transforms
import time
import datetime

def get_score(
    model: torch.nn.Module, 
    image: torch.Tensor, 
    device,
    T : int = 100,
    alpha : float = 130.0,
    r : float = 0.88,
) -> float:
    """
    Calculates the SER-FIQ score for a given aligned image using T passes.
    
    Parameters
    ----------
    model: torch.Module, Pytorch model of the network for serfiq
        Model has to have dropout enabled
    image: torch.Tensor, shape (3, h, w)
        Image, in RGB format.
    T : int, optional
        Amount of forward passes to use. The default is 100.
    alpha : float, optional
        Stretching factor, can be choosen to scale the score values
    r : float, optional
        Score displacement
    Returns
    -------
    SER-FIQ score : float
    """
    with torch.no_grad():
        repeated_image = image[None, :, :, :].to(device)
        embeddings = model(repeated_image,repeat_before_dropout=T)
        norm = torch.nn.functional.normalize(embeddings,dim=1)
        eucl_dist = torch.cdist(norm, norm)
        idx = torch.triu_indices(T,T,1)
        eucl_dist_triu = eucl_dist[idx[0],idx[1]]
        eucl_dist_mean = torch.mean(eucl_dist_triu)
        score = 2*(1 / ( 1 + eucl_dist_mean.exp() ))
        score = 1 / (1 + torch.exp( - (alpha * (score - r))))
    return score.item()

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
    for m in model.modules():
      if m.__class__.__name__.startswith('Dropout'):
        m.train()
    print(": done")

    print("Loading dataset", end="")
    dataset_eval = IrisVerificationDataset(args.data_path, transform=transforms.Compose([
        transforms.Resize(112),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]))
    print(": done")

    print("Generating scores")
    start = time.time()
    score_dict = {}
    for bn in range(len(dataset_eval)):
        image, target = dataset_eval[bn]
        time_passed = datetime.timedelta(seconds=int(time.time() - start))
        time_est = None if bn == 0 else (time_passed/(bn) * len(dataset_eval)) - time_passed
        print(f"Processed: {bn/len(dataset_eval)*100:.2f}%, est: {time_est} [h:m:s]\r", end="")
        score = get_score(model, image, device)
        score_dict[target] = score

        #if bn%args.save_period == 0:
        #    print(f"\nSaving embeddings to score_{bn}.pth")
        #    savepoint = {"scores":score_dict}
        #    torch.save(savepoint, f"score_{bn}.pth")

    print(f"\nSaving embeddings to {args.output}")
    savepoint = {"scores":score_dict}
    torch.save(savepoint, args.output)


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="ArcFace model evaluation :get: embeddings", add_help=add_help)

    parser.add_argument("--data-path", default=os.path.join("..", "Datasets", "iris_verification_NDCSI2013_01_05"), type=str, help="dataset path")
    parser.add_argument("--model", default="iresnet18", type=str, help="model name")
    parser.add_argument("--model-path", default=os.path.join("..","models","iresnet18","model_47.pth"), type=str, help="path to model .pth file")
    parser.add_argument("--embedding-size", default=128, type=int, help="size of emdedding space (default: 128)")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu Default: cpu)")
    parser.add_argument(
        "-o", "--output", default="scores.pth", type=str, help="output scores file (default: scores.pth)"
    )
    parser.add_argument(
        "-p", "--save-period", default=512, type=int, help="Periodical scores save (default: 500)"
    )

    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
    

