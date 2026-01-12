import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from trained_models.MNIST.model4 import model4
from sklearn.model_selection import train_test_split
from torch.linalg import norm
import torch.nn as nn
from utils import trainer as t
from copy import deepcopy
from typing import List

workspaces_path= os.getenv('PYTHONPATH')
print(f"Current Path: {workspaces_path}")

# --- Optimized Dataset Class ---
class OneHotVectorizedMNIST(Dataset):
  
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.num_classes = 10

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        # Get the original image (tensor) and label (int)
        image, label = self.mnist_dataset[idx]
        
        # 1. Vectorize the image: Flatten 1x28x28 to 784
        # We assume the image tensor is already normalized (C, H, W) -> (1, 28, 28)
        vectorized_image = image.flatten() 

        # 2. One-hot encode the label
        one_hot_label = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        
        return vectorized_image, one_hot_label

def apply_per_class_limit(base_dataset, n):

    indices_by_class: List[List[int]] = [[] for _ in range(10)]
    
    for idx, (_, label) in enumerate(base_dataset):
        indices_by_class[label].append(idx)
        
    all_selected_indices = []
    
    for class_indices in indices_by_class:
        m = min(n, len(class_indices)) 
        selected_indices = class_indices[:m] 
        all_selected_indices.extend(selected_indices)
        
    limited_subset = Subset(base_dataset, all_selected_indices)
    
    return limited_subset

# --- Loader Function ---
def get_loaders_vect(n_train= 20000, n_test= 10000):

    SEED = 5700
    torch.manual_seed(SEED)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    path = workspaces_path + '/trained_models/MNIST/data'  

    mnist_trainset_base = torchvision.datasets.MNIST(
        root=path, train=True, download=True, transform=transform
    )
    limited_train_base = apply_per_class_limit(mnist_trainset_base, n_train // 10)
    
    full_train_set = OneHotVectorizedMNIST(limited_train_base)
    
    
    train_indices, val_indices = train_test_split(
        range(len(full_train_set)), 
        train_size=0.8, 
        random_state=5700 # Use a fixed seed for reproducibility
    )

    trainset = Subset(full_train_set, train_indices)
    valset = Subset(full_train_set, val_indices)

    mnist_testset_base = torchvision.datasets.MNIST(
        root=path, train=False, download=True, transform=transform
    )

    limited_test_base = apply_per_class_limit(mnist_testset_base, n_train // 10)
    
    testset = OneHotVectorizedMNIST(limited_test_base)

    trainloader = DataLoader(
        trainset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )
    valloader = DataLoader(
        valset, batch_size=100, shuffle=False, num_workers=1, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )

    print("Data Loaders (Vectorized/One-Hot Optimized) created successfully.")
    return trainloader, valloader, testloader


# ACCESS LOADERS
def get_loaders():
    SEED = 5700
    torch.manual_seed(SEED)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Mean and standard deviation for MNIST
        ])

    path= workspaces_path + '/trained_models/MNIST/data' 
    trainset = torchvision.datasets.MNIST(root= path, train=True, download=True, transform=transform)
    
    indices = list(range(len(trainset)))
    train_indices, val_indices = train_test_split(indices, train_size=0.8, random_state=SEED)
    
    trainset = Subset(trainset, train_indices)
    valset = Subset(trainset, val_indices)
    
    trainloader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    valloader = DataLoader(valset, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)
    
    testset = torchvision.datasets.MNIST(root= path, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, valloader, testloader

#GET NET
def get_untrained_net():
    net= model4.Net()
    return net

def train_net(force_train=False, fn=None, kwargs={}):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    net = get_untrained_net()
    init_net = deepcopy(net)
    trainloader, valloader, testloader = get_loaders()
    model_dir= os.path.join(workspaces_path,'trained_models', 'MNIST', 'model4', 'nn_models/')
    print("Model Directory:", model_dir)   
    path_exists = os.path.exists(model_dir + 'mnist_gcnn_trained_nn.pth')
    
    if path_exists:
        checkpoint = torch.load(model_dir+'mnist_gcnn_trained_nn_0.pth', weights_only=True)
        init_net.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(model_dir+'mnist_gcnn_trained_nn.pth', weights_only=True)
        net.load_state_dict(checkpoint['state_dict'])  # Access the 'state_dict' within the loaded dictionary
        print("Model weights loaded successfully.")  
        
    if not path_exists or force_train:
        t.train_network(trainloader, valloader, testloader,
                        num_classes=10, root_path= model_dir, 
                        optimizer=torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.5),
                        lfn=  nn.NLLLoss(), 
                        num_epochs = 10,
                        name='mnist_gcnn', net=net, init_net= init_net, save_init= not force_train, fn=fn, kwargs=kwargs)
        
    return trainloader, valloader, testloader, init_net, net

def main():
    train_net()

if __name__ == "__main__":
    #For some reason executing through console adds 4sec delay
    main()
