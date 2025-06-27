import sys
import os
current_dir = os.getcwd()
print(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
print(parent_dir)
model_dir = os.path.join(parent_dir, 'trained_models', 'MNIST', 'model2', 'nn_models\\')
print(model_dir)
'''
parent_dir='C:\\Users\\garav\\AGOP\\DLR'
model_dir= 'C:\\Users\\garav\\AGOP\\DLR\\trained_models\\MNIST\\model2\\nn_models\\'
'''
sys.path.append(parent_dir)


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import torch.backends.cudnn as cudnn
from trained_models.MNIST.model2 import model2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.linalg import norm
from torchvision import models
import torch.nn as nn
from utils import trainer as t


# ACCESS LOADERS
def get_loaders():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Mean and standard deviation for MNIST
        ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainset, valset = train_test_split(trainset, train_size=0.8)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                                shuffle=False, num_workers=1)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader

#GET NET
def get_untrained_net():
    net= model2.ConvNet()
    return net

def train_net(): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_untrained_net()
    trainloader, valloader, testloader = get_loaders()
    if os.path.exists(model_dir+'mnist_conv_trained_nn.pth'):
        checkpoint = torch.load(model_dir+'mnist_conv_trained_nn.pth', map_location=torch.device(device))
        net.load_state_dict(checkpoint['state_dict'])  # Access the 'state_dict' within the loaded dictionary
        print("Model weights loaded successfully.")    
        
    t.train_network(trainloader, valloader, testloader,
                    num_classes=10, root_path= model_dir, 
                    optimizer=torch.optim.SGD(net.parameters(), lr=.1),
                    lfn=  nn.MSELoss(), 
                    num_epochs = 2,
                    name='mnist_conv', net=net)

def main():
    train_net()

if __name__ == "__main__":
    #For some reason executing through console adds 4sec delay
    main()