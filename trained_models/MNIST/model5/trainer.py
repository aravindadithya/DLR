import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import torch.backends.cudnn as cudnn
import random
import torch.backends.cudnn as cudnn
from trained_models.MNIST.model5 import model5
import numpy as np
from sklearn.model_selection import train_test_split
from torch.linalg import norm
from torchvision import models
import torch.nn as nn
from utils import trainer as t
from copy import deepcopy

#from __future__ import print_function
import argparse
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


workspaces_path= os.getenv('PYTHONPATH')
print(f"Current Path: {workspaces_path}")

# ACCESS LOADERS
def get_loaders():
    SEED = 5700
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Mean and standard deviation for MNIST
        ])
    
    path= workspaces_path + '/trained_models/MNIST/data' 
    trainset = torchvision.datasets.MNIST(root= path, train=True, download=True, transform=transform)
    trainset, valset = train_test_split(trainset, train_size=0.8)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                                shuffle=False, num_workers=1, pin_memory=True)
    
    testset = torchvision.datasets.MNIST(root= path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, valloader, testloader

#GET NET
def get_untrained_net():
    net= model5.ConvNet()
    return net

def train_net(force_train=False, fn=None, kwargs={}): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_untrained_net()
    init_net = deepcopy(net)
    trainloader, valloader, testloader = get_loaders()
    model_dir= os.path.join(workspaces_path,'trained_models', 'MNIST', 'model5', 'nn_models/')   
    path_exists = os.path.exists(model_dir + 'mnist_conv_trained_nn.pth')

    if path_exists:
        checkpoint = torch.load(model_dir+'mnist_conv_trained_nn_0.pth', weights_only=True)
        init_net.load_state_dict(checkpoint['state_dict']) 
        checkpoint = torch.load(model_dir+'mnist_conv_trained_nn.pth', weights_only=True)
        net.load_state_dict(checkpoint['state_dict'])  # Access the 'state_dict' within the loaded dictionary
        print("Model weights loaded successfully.")    

    if not path_exists or force_train:
        t.train_network(trainloader, valloader, testloader,
                        num_classes=10, root_path= model_dir, 
                        optimizer=torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.5),
                        lfn=  nn.NLLLoss(), 
                        num_epochs = 10,
                        name='mnist_conv', net=net, init_net= init_net, save_init= not force_train, fn=fn, kwargs=kwargs)
    return trainloader, valloader, testloader, init_net, net

def main():
    train_net()

if __name__ == "__main__":
    #For some reason executing through console adds 4sec delay
    main()
