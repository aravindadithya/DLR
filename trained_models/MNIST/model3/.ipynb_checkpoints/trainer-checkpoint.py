import sys
parent_dir='C:\\Users\\garav\\AGOP\\DLR'
model_dir= 'C:\\Users\\garav\\AGOP\\DLR\\trained_models\\MNIST\\model3\\nn_models\\'
sys.path.append(parent_dir)
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import torch.backends.cudnn as cudnn
import random
import torch.backends.cudnn as cudnn
from trained_models.MNIST.model3 import model3
import numpy as np
from sklearn.model_selection import train_test_split
from torch.linalg import norm
from torchvision import models
import torch.nn as nn
from utils import trainer as t


#device='cpu'
#print(f"Using device: {device}")

#TRANSFORM UTIL FUNCTIONS
def one_hot_data(dataset, num_samples=-1):
    labelset = {}
    for i in range(10):
        one_hot = torch.zeros(10)
        one_hot[i] = 1
        labelset[i] = one_hot

    subset = [(ex.flatten(), labelset[label]) for \
              idx, (ex, label) in enumerate(dataset) if idx < num_samples]
    return subset


def group_by_class(dataset):
    labelset = {}
    for i in range(10):
        labelset[i] = []
    for i, batch in enumerate(dataset):
        img, label = batch
        labelset[label].append(img.view(1, 3, 32, 32))
    return labelset


def split(trainset, p=.8):
    train, val = train_test_split(trainset, train_size=p)
    return train, val

def merge_data(mnist, n):
    #cifar_by_label = group_by_class(cifar)

    mnist_by_label = group_by_class(mnist)

    data = []
    labels = []

    labelset = {}

    for i in range(10):
        one_hot = torch.zeros(1, 10)
        one_hot[0, i] = 1
        labelset[i] = one_hot

    for l in mnist_by_label:

        #cifar_data = torch.cat(cifar_by_label[l])
        mnist_data = torch.cat(mnist_by_label[l])
        min_len = len(mnist_data)
        m = min(n, min_len)
        #cifar_data = cifar_data[:m]
        mnist_data = mnist_data[:m]

        merged = torch.cat([mnist_data], axis=-1)
        #for i in range(3):
           # vis.image(merged[i])
        data.append(merged.reshape(m, -1))
        print(merged.shape)
        labels.append(np.repeat(labelset[l], m, axis=0))
    data = torch.cat(data, axis=0)

    labels = np.concatenate(labels, axis=0)
    merged_labels = torch.from_numpy(labels)

    return list(zip(data, labels))

# ACCESS LOADERS
def get_loaders():
    SEED = 5700
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    #cudnn.benchmark = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
            [transforms.ToTensor()
            ])
    
    def repeat_channel(x):
        return x.repeat(3, 1, 1)
    
    mnist_transform = transforms.Compose(
        [transforms.Resize([32, 32]),
         transforms.ToTensor(),
         transforms.Lambda(repeat_channel)]
    )
    
    path= './data'  
        
    mnist_trainset = torchvision.datasets.MNIST(root=path,
                                                    train=True,
                                                    transform=mnist_transform,
                                                    download=True)
    
    #trainset = group_by_class(mnist_trainset)
    trainset = merge_data(mnist_trainset, 5000)
    trainset, valset = split(trainset, p=.8)
    print("Train Size: ", len(trainset), "Val Size: ", len(valset))
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                                  shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                                shuffle=False, num_workers=1)
    
    
    mnist_testset = torchvision.datasets.MNIST(root=path,
                                                   train=False,
                                                   transform=mnist_transform,
                                                   download=True)
    
    print("Test Size: ", len(mnist_testset))
    testset = merge_data(mnist_testset, 900)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                 shuffle=False, num_workers=2)

    return trainloader, valloader, testloader

#GET NET
def get_untrained_net():
    net = model3.Net(3072, num_classes=10)
    return net

def train_net(): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_untrained_net()
    trainloader, valloader, testloader = get_loaders()
    if os.path.exists(model_dir+'mnist_fc_trained_nn.pth'):
        checkpoint = torch.load(model_dir+'mnist_fc_trained_nn.pth', map_location=torch.device(device))
        net.load_state_dict(checkpoint['state_dict'])  # Access the 'state_dict' within the loaded dictionary
        print("Model weights loaded successfully.")
    
    t.train_network(trainloader, valloader, testloader,
                    num_classes=10, root_path= model_dir, 
                    optimizer=torch.optim.SGD(net.parameters(), lr=.1),
                    lfn=  nn.MSELoss(), 
                    num_epochs = 2,
                    name='mnist_fc', net=net)

def main():
    train_net()

if __name__ == "__main__":
    #For some reason executing through console adds 4sec delay
    main()