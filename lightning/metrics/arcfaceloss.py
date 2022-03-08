import imp
from .arcface import *

import torch
import torch.nn.functional as F
from torch import nn

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = ArcFace(s, m)
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.loss = nn.CrossEntropyLoss()
        nn.init.xavier_uniform_(self.weight)
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        fc = self.fc(cosine, label)
        return self.loss(fc, label), cosine
    
class CosFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = CosFace(s, m)
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.loss = nn.CrossEntropyLoss()
        nn.init.xavier_uniform_(self.weight)
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        fc = self.fc(cosine, label)
        return self.loss(fc, label), cosine

