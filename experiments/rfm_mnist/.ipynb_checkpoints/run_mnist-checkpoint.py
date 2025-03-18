import torch
import torchvision
import torchvision.transforms as transforms
import trainer as t
from torch.utils.data import Dataset
import random
import torch.backends.cudnn as cudnn
import rfm
import numpy as np
import visdom
import numpy as np
from sklearn.model_selection import train_test_split
from torch.linalg import norm

vis = visdom.Visdom('http://127.0.0.1', use_incoming_socket=False)
vis.close(env='main')

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

def main():
    SEED = 5636
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    cudnn.benchmark = True

    transform = transforms.Compose(
        [transforms.ToTensor()
        ])

    mnist_transform = transforms.Compose(
        [transforms.Resize([32,32]),
         transforms.ToTensor(),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])


    path = '~/data/'
    
    mnist_trainset = torchvision.datasets.MNIST(root=path,
                                                train=True,
                                                transform=mnist_transform,
                                                download=False)

    trainset = merge_data(cifar_trainset, mnist_trainset, 5000)
    trainset, valset = split(trainset, p=.8)
    print("Train Size: ", len(trainset), "Val Size: ", len(valset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                            shuffle=False, num_workers=1)


    mnist_testset = torchvision.datasets.MNIST(root=path,
                                               train=False,
                                               transform=mnist_transform,
                                               download=False)

    print("Test Size: ", len(testset))

    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    name = 'mnist'
    rfm.rfm(trainloader, valloader, testloader, name=name,
            batch_size=10, iters=1, reg=1e-3)

    t.train_network(trainloader, valloader, testloader,
                    num_classes=10,
                    name=name)


if  __name__ == "__main__":
    main()