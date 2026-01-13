import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.backends.cudnn as cudnn
from trained_models.MNIST.model1 import model1
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch.nn as nn
from utils import trainer as t
from copy import deepcopy

workspaces_path= os.getenv('PYTHONPATH')
print(f"Current Path: {workspaces_path}")

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


class MNISTDataset(torch.utils.data.Dataset):
    """Lazy dataset - just applies transforms on-demand"""
    def __init__(self, mnist_dataset, n_per_class=500):
        self.dataset = mnist_dataset
        # Limit to n_per_class*10 samples (fast - no iteration)
        self.limit = min(n_per_class * 10, len(self.dataset))
    
    def __len__(self):
        return self.limit
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # img is already (3, 32, 32) from the transform pipeline
        img_flat = img.flatten()  # (3072,)
        
        # One-hot encode label
        one_hot = torch.zeros(10, dtype=torch.float32)
        one_hot[label] = 1.0
        
        return img_flat, one_hot

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
    
    path= workspaces_path + '/trained_models/MNIST/data'  
        
    mnist_trainset = torchvision.datasets.MNIST(root=path,
                                                    train=True,
                                                    transform=mnist_transform,
                                                    download=True)
    
    trainset = MNISTDataset(mnist_trainset, n_per_class=500)
    indices = list(range(len(trainset)))
    train_indices, val_indices = train_test_split(indices, train_size=0.8, random_state=SEED)
    
    trainset = Subset(trainset, train_indices)
    valset = Subset(trainset, val_indices)
    print("Train Size: ", len(trainset), "Val Size: ", len(valset))
    
    trainloader = DataLoader(trainset, batch_size=100,
                                                  shuffle=True, num_workers=2, pin_memory=True)
    valloader = DataLoader(valset, batch_size=100,
                                                shuffle=False, num_workers=1, pin_memory=True)
    
    
    mnist_testset = torchvision.datasets.MNIST(root=path,
                                                   train=False,
                                                   transform=mnist_transform,
                                                   download=True)
    
    testset = MNISTDataset(mnist_testset, n_per_class=90)
    print("Test Size: ", len(testset))
    testloader = DataLoader(testset, batch_size=128,
                                                 shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, valloader, testloader

#GET NET
def get_untrained_net():
    net = model1.Net(3072, num_classes=10)
    return net

def train_net(force_train=False, fn=None, kwargs={}): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_untrained_net()
    init_net = deepcopy(net)
    trainloader, valloader, testloader = get_loaders()
    model_dir = os.path.join(workspaces_path,'trained_models', 'MNIST', 'model1', 'nn_models/')
    path_exists = os.path.exists(model_dir +'mnist_fc_trained_nn.pth')
    
    if path_exists:
        checkpoint = torch.load(model_dir +'mnist_fc_trained_nn.pth', weights_only=True)
        net.load_state_dict(checkpoint['state_dict'])  # Access the 'state_dict' within the loaded dictionary
        checkpoint = torch.load(model_dir+'mnist_fc_trained_nn_0.pth', weights_only=True)
        init_net.load_state_dict(checkpoint['state_dict'])
        print("Model weights loaded successfully.")  
        
    if not path_exists or force_train:   
        t.train_network(trainloader, valloader, testloader,
                        num_classes=10, root_path= model_dir, 
                        optimizer=torch.optim.SGD(net.parameters(), lr=.1),
                        lfn=  nn.MSELoss(), 
                        num_epochs = 10,
                        name='mnist_fc', net=net, init_net= init_net, save_init= not force_train, fn=fn, kwargs=kwargs)  
       
        
    return trainloader, valloader, testloader, init_net, net
    

def main():
    train_net()

if __name__ == "__main__":
    #For some reason executing through console adds 4sec delay
    main()