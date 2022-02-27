import torch
import torch.nn.functional as F
from torch import nn

class PassthroughDecoder(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(PassthroughDecoder, self).__init__()
        assert in_features == out_features, "Input feature count must equal output feature count for passtrough"
        self.in_features = in_features
        self.out_features = out_features
    def forward(self, input, label):
        return input

