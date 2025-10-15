import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import torch.backends.cudnn as cudnn
import random
from trained_models.CIFAR.model2 import model2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.linalg import norm
from torchvision import models
import torch.nn as nn
from utils import trainer as t

#from __future__ import print_function
import argparse
from torchvision import datasets
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from copy import deepcopy

# ACCESS LOADERS
def get_loaders():
    
    SEED = 5700
    means = (0.4914, 0.4822, 0.4465)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='/work/DLR/trained_models/CIFAR/data', train=True, download=True, transform=transform_train)
    trainset, valset = train_test_split(trainset, train_size=0.8)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                                shuffle=False, num_workers=1, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root='/work/DLR/trained_models/CIFAR/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, valloader, testloader

#GET NET
def get_untrained_net():
    net= model2.ResNet34()
    return net

def train_net(force_train=False, num_epochs=10): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_untrained_net()
    init_net = deepcopy(net)
    trainloader, valloader, testloader = get_loaders()
    model_dir= os.path.join('/work/DLR','trained_models', 'CIFAR', 'model2', 'nn_models/')   
    init_path_exists = os.path.exists(model_dir + 'cifar_gcnn_init_nn.pth')
    path_exists = os.path.exists(model_dir + 'cifar_gcnn_trained_nn.pth')
    
    if init_path_exists and path_exists:
        checkpoint = torch.load(model_dir+'cifar_gcnn_trained_nn.pth', weights_only=True)
        init_net.load_state_dict(checkpoint['state_dict'])     
        checkpoint = torch.load(model_dir+'cifar_gcnn_trained_nn.pth', weights_only=True)
        net.load_state_dict(checkpoint['state_dict'])  # Access the 'state_dict' within the loaded dictionary
        print("Model weights loaded successfully.")  
        if force_train:
            t.train_network(trainloader, valloader, testloader,
                        num_classes=10, root_path= model_dir, 
                        optimizer=torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.5),
                        lfn=  nn.CrossEntropyLoss(), 
                        num_epochs = num_epochs,
                        name='cifar_gcnn', net=net)
            
    elif not path_exists and not init_path_exists:
        t.train_network(trainloader, valloader, testloader,
                        num_classes=10, root_path= model_dir, 
                        optimizer=torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.5),
                        lfn=  nn.CrossEntropyLoss(), 
                        num_epochs = num_epochs,
                        name='cifar_gcnn', net=net)   
        d = {}
        d['state_dict'] = init_net.state_dict()
        torch.save(d, model_dir + 'cifar_gcnn_init_nn.pth')
    else:
        print("Error. Try deleting all the weight files to start a fresh training")

    return trainloader, valloader, testloader, init_net, net

def main():
    train_net()

if __name__ == "__main__":
    #For some reason executing through console adds 4sec delay
    main()
