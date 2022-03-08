import torch
import torch.nn.functional as F
from torch import nn

class PassthroughLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(PassthroughLoss, self).__init__()
        assert in_features == out_features, "Input feature count must equal output feature count for passtrough"
        self.in_features = in_features
        self.out_features = out_features
        self.loss = nn.CrossEntropyLoss()
    def forward(self, input, label):
        return self.loss(input, label), input

