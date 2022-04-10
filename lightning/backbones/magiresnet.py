import torch
from torch import nn
# from torchvision.models.utils import load_state_dict_from_url

__all__ = ['magiresnet18', 'magiresnet34', 'magiresnet50', 'magiresnet100']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MagIBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(MagIBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-05, momentum=0.9)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class MagIResNet(nn.Module):
    fc_scale = 14 * 14

    def __init__(self, block, layers, in_channels=3, num_classes=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, 
                 dropout_prob0=0.0):
        super(MagIResNet, self).__init__()
        self.in_channels = in_channels
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.dropout_prob0 = dropout_prob0
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn2 = nn.BatchNorm2d(
            512 * block.expansion, eps=2e-05, momentum=0.9)
        if self.dropout_prob0 != 0:
            self.dropout = nn.Dropout2d(p=dropout_prob0, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, MagIBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion,
                               eps=2e-05, momentum=0.9),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x, repeat_before_dropout:int = 1):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = x.repeat(repeat_before_dropout, 1, 1, 1)
        if self.dropout_prob0 != 0:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _magiresnet(block, layers, pretrained, progress, **kwargs):
    model = MagIResNet(block, layers, **kwargs)
    return model


def magiresnet18(pretrained=False, progress=True, **kwargs):
    return _magiresnet(MagIBasicBlock, [2, 2, 2, 2], pretrained, progress,
                    **kwargs)


def magiresnet34(pretrained=False, progress=True, **kwargs):
    return _magiresnet(MagIBasicBlock, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)


def magiresnet50(pretrained=False, progress=True, **kwargs):
    return _magiresnet(MagIBasicBlock, [3, 4, 14, 3], pretrained, progress,
                    **kwargs)


def magiresnet100(pretrained=False, progress=True, **kwargs):
    return _magiresnet(MagIBasicBlock, [3, 13, 30, 3], pretrained, progress,
                    **kwargs)
