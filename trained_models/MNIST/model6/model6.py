import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
from torchvision import models
from torch.nn.functional import upsample
from copy import deepcopy
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=3, stride=(1,1), padding=(0,0)),
            nn.ReLU(),
            nn.Conv2d(40, 40, kernel_size=3, stride=(1,1), padding=(0,0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(40, 80, kernel_size=3, stride=(1,1), padding=(0,0)),
            nn.ReLU(),
            nn.Conv2d(80, 80, kernel_size=3, stride=(1,1), padding=(0,0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(80 * 4 * 4, 50),
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
