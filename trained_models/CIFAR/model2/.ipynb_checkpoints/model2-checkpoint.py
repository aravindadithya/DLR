'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import sys
import os
current_dir = os.getcwd()
#print(current_dir)
parent_dir = os.path.join(current_dir, 'DLR')
#print(parent_dir)
model_dir = os.path.join(parent_dir, 'trained_models', 'CIFAR', 'model2', 'nn_models/')
#print(model_dir)


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from utils.groupy.gconv.pytorch_gconv.splitgconv2d import P4MConvZ2, P4MConvP4M

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.features= nn.Sequential(
        P4MConvP4M(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(planes),
        nn.ReLU(),
        P4MConvP4M(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(planes)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4MConvP4M(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )
        self.lrelu = nn.ReLU()

    def forward(self, x):
        out = self.features(x)
        out += self.shortcut(x)
        out = self.lrelu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.features = nn.Sequential(
        P4MConvP4M(in_planes, planes, kernel_size=1, bias=False),
        nn.BatchNorm3d(planes),
        nn.ReLU(),
        P4MConvP4M(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(planes),
        nn.ReLU(),
        P4MConvP4M(planes, self.expansion*planes, kernel_size=1, bias=False),
        nn.BatchNorm3d(self.expansion*planes)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4MConvP4M(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )
        self.lrelu = nn.ReLU()

    def forward(self, x):
        out = self.features(x)
        out += self.shortcut(x)
        out = self.lrelu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 23
        self.layers = [       
        P4MConvZ2(3, 23, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(23),
        nn.ReLU()] + self._make_layer(block, 23, num_blocks[0], stride=1) + self._make_layer(block, 45, num_blocks[1], stride=2) + self._make_layer(block, 91, num_blocks[2], stride=2) + self._make_layer(block, 181, num_blocks[3], stride=2)
        self.features = nn.Sequential(*self.layers)
        
        self.classifier = nn.Linear(181*8*block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        #return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        out = self.features(x)
        outs = out.size()
        out = out.view(outs[0], outs[1]*outs[2], outs[3], outs[4])
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()