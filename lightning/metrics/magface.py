#!/usr/bin/env python
import sys
sys.path.append("..")
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import math
import torch
import torch.nn as nn

class MagFaceLoss(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        arc_scale = 64,
        l_margin = 0.45,
        u_margin = 0.8,
        l_a = 10,
        u_a = 110,
        lambda_g = 20
    ):
        super(MagFaceLoss, self).__init__()
        self.fc = MagLinear(in_features,
                            out_features,
                            scale=arc_scale)
        self.criterion = MagLoss(l_a, u_a, l_margin, u_margin)
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.l_a = l_a
        self.u_a = u_a
        self.lambda_g = lambda_g

    def _margin(self, x):
        """generate adaptive margin
        """
        margin = (self.u_margin-self.l_margin) / \
            (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin

    def forward(self, x, target):
        logits, x_norm = self.fc(x, self._margin, self.l_a, self.u_a)
        loss_id, loss_g, one_hot, output = self.criterion(logits, target, x_norm)
        loss = loss_id + self.lambda_g * loss_g
        return loss, output


class MagLinear(torch.nn.Module):
    """
    Parallel fc for Mag loss
    """

    def __init__(self, in_features, out_features, scale=64.0, easy_margin=True):
        super(MagLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn = nn.BatchNorm1d(in_features, eps=2e-05, momentum=0.9)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.scale = scale
        self.easy_margin = easy_margin

    def forward(self, x, m, l_a, u_a):
        """
        Here m is a function which generate adaptive margin
        """
        x_bn = self.bn(x)
        x_norm = torch.norm(x_bn, dim=1, keepdim=True).clamp(l_a, u_a)
        ada_margin = m(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)

        # norm the weight
        weight_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(F.normalize(x_bn), weight_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(
                cos_theta > threshold, cos_theta_m, cos_theta - mm)
        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta

        return [cos_theta, cos_theta_m], x_norm


class MagLoss(torch.nn.Module):
    """
    MagFace Loss.
    """

    def __init__(self, l_a, u_a, l_margin, u_margin, scale=64.0):
        super(MagLoss, self).__init__()
        self.l_a = l_a
        self.u_a = u_a
        self.scale = scale
        self.cut_off = np.cos(np.pi/2-l_margin)
        self.large_value = 1 << 10

    def calc_loss_G(self, x_norm):
        g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        return torch.mean(g)

    def forward(self, input, target, x_norm):
        loss_g = self.calc_loss_G(x_norm)

        cos_theta, cos_theta_m = input
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        loss = F.cross_entropy(output, target, reduction='mean')
        return loss.mean(), loss_g, one_hot, output
