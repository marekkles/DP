import torch
from torchvision import transforms

__all__ = [
    'train_transform', 'val_transform', 
    'test_transform', 'predict_transform'
]

def train_transform(
    RandomAffine, Resize, RandomAdjustSharpness, 
    RandomAutocontrast, RandomInvert, Normalize, RandomErasing
):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomAdjustSharpness(**RandomAdjustSharpness),
        transforms.RandomAutocontrast(**RandomAutocontrast),
        transforms.RandomInvert(**RandomInvert),
        transforms.RandomAffine(**RandomAffine),
        transforms.Resize(**Resize),
        transforms.Normalize(**Normalize),
        transforms.RandomErasing(**RandomErasing),
    ])
def val_transform(Resize, Normalize):
    return transforms.Compose([
        transforms.Resize(**Resize),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(**Normalize),
    ])
def test_transform(Resize, Normalize):
    return transforms.Compose([
        transforms.Resize(**Resize),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(**Normalize),
    ])

def predict_transform(Resize, Normalize):
    return transforms.Compose([
        transforms.Resize(**Resize),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(**Normalize),
    ])