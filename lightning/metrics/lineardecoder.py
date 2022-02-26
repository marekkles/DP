import torch
import torch.nn.functional as F
from torch import nn

class LinearDecoder(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(LinearDecoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine

