''' This module does the following
1. Scan the network for conv layers
2. For each FC layer compute W^TW of eq 3
3. For each FC layer compute the AGOP(AJOP in case of multiple outputs)
4. For each conv layer print the pearson correlation between 2 and 3
'''

import torch
import torch.nn as nn
import random
import numpy as np
#from functorch import jacrev
#from torch.func import jacrev
from torch.nn.functional import pad
#import dataset
from numpy.linalg import eig
from copy import deepcopy
from torch.linalg import norm, svd
from torchvision import models
#import visdom
from torch.linalg import norm, eig


SEED = 2323

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

#vis = visdom.Visdom('http://127.0.0.1', use_incoming_socket=False)
#vis.close(env='main')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device='cpu'
#print(f"Using device: {device}")

def get_jacobian(net, data):
    with torch.no_grad():
        return torch.vmap(torch.func.jacrev(net))(data).transpose(0, 2).transpose(0, 1)

def min_max(M):
    return (M - M.min()) / (M.max() - M.min())

def sqrt(G):
    U, s, Vt = svd(G)
    s = torch.pow(s, 1./2)
    G = U @ torch.diag(s) @ Vt
    return G


def correlation(M, G):
    M -= M.mean()
    G -= G.mean()
    M = M.double()
    G = G.double()
    normM = norm(M.flatten())
    normG = norm(G.flatten())

    corr = torch.dot(M.flatten(), G.flatten()) / (normM * normG)
    return corr

def egop(net, dataset, batch_size=800, cutoff=10, centering=False):
    device = torch.device('cuda')
    bs = 800
    batches = torch.split(dataset, bs)
    net = net.cuda()
    G = 0

    Js = []
    for batch_idx, data in enumerate(batches):
        data = data.to(device)
        print("Computing Jacobian for batch: ", batch_idx, len(batches))
        J = get_jacobian(net, data)
        Js.append(J.cpu())
        # Optional for stopping EGOP computation early
        if batch_idx > cutoff:
            break
    Js = torch.cat(Js, dim=-1)
    if centering:
        J_mean = torch.mean(Js, dim=-1).unsqueeze(-1)
        Js = Js - J_mean

    Js = torch.transpose(Js, 2, 0)
    Js = torch.transpose(Js, 1, 2)
    print(Js.shape)
    batches = torch.split(Js, bs)
    for batch_idx, J in enumerate(batches):
        print(batch_idx, len(batches))
        m, c, d = J.shape
        J = J.cuda()
        G += torch.einsum('mcd,mcD->dD', J, J).cpu()
        del J
    G = G * 1/len(Js)

    return G


def load_nn(net, init_net, layer_idx=0):
   
    count = 0
    
    # Get the layer_idx+1 th conv layer
    #TODO: Add functionality to access classifier layers too.
    for idx, m in enumerate(net.features):
        if isinstance(m, nn.Linear):
            count += 1
        if count-1 == layer_idx:
            l_idx = idx
            break
    
    subnet_l = deepcopy(net)
    subnet_r = deepcopy(net)
    
    # Truncate all layers before l_idx.
    subnet_r.features = net.features[l_idx:]
    subnet_l.features = net.features[:l_idx]
    
    M = net.features[l_idx].weight.data
    # Compute WW which is (s,s) matrix
    M =torch.matmul(M.T, M)
    M0 = init_net.features[l_idx].weight.data
    # Compute W0tW0 which is (s,s) matrix
    M0 =torch.matmul(M0.T, M0)
    return net, subnet_l, subnet_r, M, M0, l_idx

def get_layer_output(net, trainloader, layer_idx=0):
    print(net)
    net.cpu()
    net.eval()
    out = []
    for idx, batch in enumerate(trainloader):
        data, labels = batch
        data.cpu()
        if layer_idx == 0:
            out.append(data)
        else:
            out.append(net(data))
        '''
        elif layer_idx == 1:
            o = neural_model.Nonlinearity()(net.first(data))
            out.append(o.cpu())
        elif layer_idx > 1:
            o = net.first(data)
            for l_idx, m in enumerate(net.middle):
                o = m(o)
                if l_idx + 1 == layer_idx:
                    o = neural_model.Nonlinearity()(o)
                    out.append(o.cpu())
                    break'''
    out = torch.cat(out, dim=0)
    net.cpu()
    return out

def verify_NFA(net, init_net, trainloader, layer_idx=0, batch_size=800, cutoff=10, chunk_idx=1):
    
    # TODO: Implement chunking for AGOP
    
    net, subnet_l, subnet_r, M, M0, l_idx = load_nn(net, init_net, layer_idx=layer_idx)
    print(M.shape)
    i_val = correlation(M0.cuda(), M.cuda())
    print("Correlation between Initial and Trained CNFM: ", i_val)
    
    out = get_layer_output(subnet_l.features, trainloader, layer_idx=layer_idx)
    G = egop(subnet_r, out, batch_size, cutoff, centering=True)
    G2 = egop(subnet_r, out, batch_size, cutoff, centering=False)
    G = sqrt(G.cuda())
    G2 = sqrt(G2.cuda())
    print("Shape of grad matrix",G.shape)
    #G = sqrt(G.cuda())
    #Gop = G.clone()
    centered_correlation = correlation(M.to(device), G.to(device))
    uncentered_correlation = correlation(M.to(device), G2.to(device))
    print("Full Matrix Correlation Centered: " , centered_correlation)
    print("Full Matrix Correlation Uncentered: " , uncentered_correlation)
    #return Gop


'''
TODO:
1. Implement chunking for AGOP
2. Make the optimizing variables batch_size, cutoff consistent with the other implementation of AGOP FC
'''