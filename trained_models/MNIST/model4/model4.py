import sys
import os
current_dir = os.getcwd()
#print(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
#print(parent_dir)
model_dir = os.path.join(parent_dir, 'trained_models', 'MNIST', 'model4', 'nn_models\\')
#print(model_dir)

'''
import sys
parent_dir='C:\\Users\\garav\\AGOP\\DLR'
model_dir= 'C:\\Users\\garav\\AGOP\\DLR\\trained_models\\MNIST\\model4\\nn_models\\'
sys.path.append(parent_dir)
import os'''
sys.path.append(parent_dir)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils.groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from utils.groupy.gconv.pytorch_gconv.pooling import PlaneGroupSpatialMaxPool

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()    
        self.features = nn.Sequential(
        P4ConvZ2(1, 10, kernel_size=3),
        nn.ReLU(),
        P4ConvP4(10, 10, kernel_size=3),
        nn.ReLU(),
        PlaneGroupSpatialMaxPool(2,2),
        P4ConvP4(10, 20, kernel_size=3),  
        nn.ReLU(),
        P4ConvP4(20, 20, kernel_size=3),
        nn.ReLU(),
        PlaneGroupSpatialMaxPool(2,2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(4*4*20*4, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size()[0], -1)
        x = x.reshape(x.size()[0], -1)
        x = self.classifier(x)
        #x = F.dropout(x, training=self.training)
        return F.log_softmax(x)