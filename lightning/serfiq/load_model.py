import os
import yaml
import backbones
import torch

__all__ = ['load_model', 'setup_model']

def setup_model(model: torch.nn.Module) -> None:
    model.eval()
    for m in model.modules():
      if m.__class__.__name__.startswith('Dropout'):
        m.train()

def load_model(run_root: str) -> torch.nn.Module:
    """Load model from given run

    Args:
        run_root (str): Root directory of the run

    Returns:
        torch.nn.Module: Model
    """
    args_path = os.path.join(run_root,"args.yaml")
    with open(args_path, "r") as f:
        args_file = yaml.load(f)
    bakbone_path = os.path.join(run_root,f"encoder-{args_file['run_name']}.pth")
    model = backbones.__dict__[args_file["backbone"]](**args_file["backbone_args"])
    model.load_state_dict(bakbone_path)
    return model, args_file
