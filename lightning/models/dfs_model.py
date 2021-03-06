# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt

__all__ = ['DfsNet']

class MobileNetV2_encoder(nn.Module):
    def __init__(self, pretrained):
        super(MobileNetV2_encoder, self).__init__()
        model = torchvision.models.mobilenet_v2(pretrained=pretrained)
        self.low_feature = model.features[:5]
        self.high_feature = model.features[5:]

    def forward(self, input):
        out1 = self.low_feature(input)
        out2 = self.high_feature(out1)
        return out1, out2


class LRASPPV2(nn.Module):
    """Lite R-ASPP"""
    def __init__(self, nclass=2):
        super(LRASPPV2, self).__init__()
        self.b0 = nn.Sequential(nn.Conv2d(1280, 128, 1, bias=False),
                                nn.BatchNorm2d(128), nn.ReLU(True))
        self.b1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1280, 128, 1, bias=False),
            nn.Sigmoid(),
        )

        self.project = nn.Conv2d(128, nclass, 1)
        self.shortcut = nn.Conv2d(32, nclass, 1)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, y):
        size = x.shape[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2  # check it
        x = self.project(x)
        y = self.shortcut(y)
        out = F.adaptive_avg_pool2d(y, size) + x
        return out


class MobileNetV2_Lite(nn.Module):
    def __init__(self, pretrained=False, mask_learn_rate=0.5):
        super(MobileNetV2_Lite, self).__init__()
        self.encoder = MobileNetV2_encoder(pretrained)

        self.decoder = LRASPPV2()

        self.linear = nn.Sequential(
            nn.Linear(1280, 512, True),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 64, True),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.init_params(self.linear)

        for p in self.parameters():
            p.requires_grad = True
        if mask_learn_rate == 1:
            for p in self.linear.parameters():
                p.requires_grad = False
        elif mask_learn_rate == 0:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False

    def init_params(self, target):
        for m in target:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        low_fea, high_fea = self.encoder(input)

        mask = self.decoder(high_fea, low_fea)
        att_mask = torch.unsqueeze(torch.softmax(mask, 1)[:, 1, :, :], 1)
        out_mask = nn.functional.interpolate(
            mask, (input.shape[2] // 4, input.shape[3] // 4),
            mode='bilinear',
            align_corners=True)

        pred = torch.sum(high_fea * att_mask,
                         dim=(2, 3)) / (torch.sum(att_mask, dim=(2, 3)) + 1e-8)
        pred = pred.view(pred.size(0), -1)
        pred = self.linear(pred)
        return pred, out_mask


class DfsNet(pl.LightningModule):
    def __init__(
        self,
        mobilenet_checkpoint: str
    ):
        super().__init__()
        self.mobilenet_checkpoint = mobilenet_checkpoint
        self.mobilenet_state_dict = torch.load(
            self.mobilenet_checkpoint, 
            map_location=torch.device('cpu')
        )
        self.mobilenet = MobileNetV2_Lite()
        self.mobilenet.load_state_dict(self.mobilenet_state_dict['model'])

        self.save_hyperparameters()
    def forward(self, x):
        pred, out_mask = self.mobilenet(x)
        return pred, out_mask
    def predict_step(self, batch, batch_idx):
        x, y = batch
        quality, _ = self(x)
        return  {"quality" : quality, "label": y}

