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
#from functorch import jacrev, vmap
from torch.func import jacrev
from torch.nn.functional import pad
#import dataset
from numpy.linalg import eig
from copy import deepcopy
from torch.linalg import norm, svd
from torchvision import models
import visdom
from torch.linalg import norm, eig


SEED = 2323

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

vis = visdom.Visdom('http://127.0.0.1', use_incoming_socket=False)
vis.close(env='main')

def get_jacobian(net, data, c_idx=0, chunk=100):
    with torch.no_grad():
        def single_net(x):
            # x is (s)
            return net(x.unsqueeze(0))[:,c_idx*chunk:(c_idx+1)*chunk].squeeze(0)
        # Parallelize across the images.
        return torch.vmap(jacrev(single_net))(data) #(n,chunk,s)

def min_max(M):
    return (M - M.min()) / (M.max() - M.min())

def sqrt(G):
    U, s, Vt = svd(G)
    s = torch.pow(s, 1./2)
    G = U @ torch.diag(s) @ Vt
    return G


def correlation(M1, M2):
    M1 -= M1.mean()
    M2 -= M2.mean()

    norm1 = norm(M1.flatten())
    norm2 = norm(M2.flatten())

    return torch.sum(M1.cuda() * M2.cuda()) / (norm1 * norm2)

def egop(model, z, c=10, chunk_idxs=1):
    ajop = 0
    #Chunking is done to compute jacobian as chunks. This saves memory
    chunk = c // chunk_idxs
    for i in range(chunk_idxs):
        grads = get_jacobian(model, z, c_idx=i, chunk=chunk) #(n,chunk,s)
        grads_t = grads.transpose(1, 2) 
        ajop_matmul= torch.matmul(grads_t, grads) #(n,s,s)
        #Clarify: mean and sum are making no difference here. Check if trainloader has grouped images
        ajop += torch.mean(ajop_matmul, dim=0) #(s,s)
    return ajop



def get_grads(net, patchnet, trainloader, max_batch, classes, chunk_idx,
              kernel=(3,3), padding=(1,1),
              stride=(1,1), layer_idx=0):
    net.eval()
    net.cuda()
    patchnet.eval()
    patchnet.cuda()
    M = 0
    #M.cuda()
    
    # Num images for taking AGOP (Can be small for early layers)
    MAX_NUM_IMGS = max_batch

    for idx, batch in enumerate(trainloader):
        print("Computing GOP for sample " + str(idx) + \
              " out of " + str(MAX_NUM_IMGS))
        imgs, _ = batch
        #imgs=imgs[:]
        with torch.no_grad():
            imgs = imgs.cuda()        
            # Run the first half of the network wrt to the current layer 
            ip = net.features[:layer_idx](imgs).cpu() #(n,s)
            
        #print(patches.shape)
        M += egop(patchnet,ip.cuda(), classes, chunk_idx).cuda()
        del imgs
        torch.cuda.empty_cache()
        if idx >= MAX_NUM_IMGS:
            break
    net.cpu()
    patchnet.cpu()
    return M

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
    
    patchnet = deepcopy(net)
    
    # Truncate all layers before l_idx.
    patchnet.features = net.features[l_idx:]
    
    M = net.features[l_idx].weight.data
    # Compute WW which is (s,s) matrix
    M =torch.matmul(M.T, M)
    M0 = init_net.features[l_idx].weight.data
    # Compute W0tW0 which is (s,s) matrix
    M0 =torch.matmul(M0.T, M0)
    return net, patchnet, M, M0, l_idx


def verify_NFA(net, init_net, trainloader, layer_idx=0, max_batch=10, classes=10, chunk_idx=1):


    net, patchnet, M, M0, l_idx = load_nn(net, init_net, layer_idx=layer_idx)

    i_val = correlation(M0.cuda(), M.cuda())
    print("Correlation between Initial and Trained CNFM: ", i_val)

    G = get_grads(net, patchnet, trainloader,  max_batch, classes, chunk_idx,
                  layer_idx=l_idx)
    print("Shape of grad matrix",G.shape)
    G = sqrt(G.cuda())
    Gop = G.clone()
    r_val = correlation(M.cuda(), G.cuda())
    print("Correlation between Trained CNFM and AGOP: ", r_val)
    print("Final: ", i_val, r_val)
    return Gop

def vis_transform_image(net, img, G, layer_idx=0):
   #TODO: What to visualise for the FC layers?
    count = -1
    
    # Computes WtW for the weights(ignoring its bias) of layer_idx+1 the conv layer
    for idx, p in enumerate(net.parameters()):
        if len(p.shape) > 1:
            count += 1
        if count == layer_idx:
            M = p.data
            _, ki, q, s = M.shape

            M = M.reshape(-1, ki*q*s)
            M = torch.einsum('nd, nD -> dD', M, M)
            break

    count = 0
    l_idx = None
    
    # Get the layer_idx+1 conv layer 
    for idx, m in enumerate(net.features):
        if isinstance(m, nn.Conv2d):
            print(m, count)
            count += 1

        if count-1 == layer_idx:
            l_idx = idx
            break

    net.eval()
    net.cuda()
    img = img.cuda()
    img = net.features[:l_idx](img).cpu()
    net.cpu()
    
    # If G is given which is expected to be the AGOP of layer_idx+1 conv layer then that is used.
    if G is not None:
        M = G

    patches = patchify(img, (q, s), (1, 1))
    
    print(patches.shape)
    # Patches should will be of the shape (n,w,h,c,q,s) not (n,w,h,q,s,c)
    n, w, h, q, s, c = patches.shape
    # Vectorize each patch
    patches = patches.reshape(n, w, h, q*s*c)
    # Apply either WtW or AGOP of the layer_idx+1 conv to each patch. D is c*q*s vector
    M_patch = torch.einsum('nwhd, dD -> nwhD', patches, M) #(n,w,h,c*q*s)
    
    M_patch = norm(M_patch, dim=-1) #(n,w,h)

    vis.image(min_max(M_patch[0])) #(w,h) image.


