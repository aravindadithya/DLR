''' This module does the following
1. Scan the network for conv layers
2. For each conv layer compute W^TW of eq 3
3. For each conv layer compute the AGOP(AJOP in case of multiple outputs)
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


def patchify(x, patch_size, stride_size, padding=None, pad_type='zeros'):
    '''
        Given an input image (n,c,h,w) generate (n,h_out,w_out,c,q,s) respecting stride,padding, 
        w_out is number of pathces along the width for the given stride after padding
        h_out is number of pathces along the height for the given stride after padding
        (q,s) is the kernel dimensions 
    '''
    q1, q2 = patch_size
    s1, s2 = stride_size

    if padding is None:
        pad_1 = (q1-1)//2
        pad_2 = (q2-1)//2
    else:
        pad_1, pad_2 = padding

    pad_dims = (pad_2, pad_2, pad_1, pad_1)
    if pad_type == 'zeros':
        x = pad(x, pad_dims)
    elif pad_type == 'circular':
        x = pad(x, pad_dims, 'circular')
        
    patches = x.unfold(2, q1, s1).unfold(3, q2, s2) #(n, c, h_out, w_out, q, s)
    patches = patches.transpose(1, 3).transpose(1, 2) #(n,h_out,w_out,c,q,s) 
    return patches

class PatchConvLayer(nn.Module):
    def __init__(self, conv_layer):
        super().__init__()
        self.layer = conv_layer #(k,c,q,s)

    def forward(self, patches):
        #Todo: 1. Check why the format is nwhcqr when the patches after patchify is nhwcqr.
        #      2. Why does this output n,k,w,h when the standard format is n,k,h,w
        out = torch.einsum('nwhcqr, kcqr -> nwhk', patches, self.layer.weight)
        n, w, h, k = out.shape
        out = out.transpose(1, 3).transpose(2, 3) #Should be (n,k,h_out,w_out) even though w and h are swapped
        return out

def get_jacobian(net, data, c_idx=0, chunk=100):
    with torch.no_grad():
        def single_net(x):
            # x is (h_out,w_out,c,q,s)
            return net(x.unsqueeze(0))[:,c_idx*chunk:(c_idx+1)*chunk].squeeze(0)
        # Parallelize across the images.
        return torch.vmap(jacrev(single_net))(data) #(n, chunk, h_out, w_out, c, q, s)

def egop(model, z):
    ajop = 0
    c = 10
    chunk_idxs = 1
    #Chunking is done to compute jacobian as chunks. This saves memory
    #TODO: chunk should be passed as argument
    chunk = c // chunk_idxs
    for i in range(chunk_idxs):
        J = get_jacobian(model, z, c_idx=i, chunk=chunk)
        n, c, w, h, _, _, _ = J.shape
        J = J.transpose(1, 3).transpose(1, 2) #(n, h_out, w_out, chunk, c, q, s)
        grads = J.reshape(n*w*h, c, -1) #(n*w_out*h_out, chunk, c*q*s)
        #Clarify: Where is mean taken
        ajop += torch.einsum('ncd, ncD -> dD', grads, grads) #(c*q*s,c*q*s)
    return ajop


def load_nn(net, init_net, layer_idx=0):
   
    count = 0
    
    # Get the layer_idx+1 th conv layer
    for idx, m in enumerate(net.features):
        if isinstance(m, nn.Conv2d):
            count += 1
        if count-1 == layer_idx:
            l_idx = idx
            break
    
    patchnet = deepcopy(net)
    layer = PatchConvLayer(net.features[l_idx])
    
    # Extract all the meta info of the current conv layer.
    (q, s) = net.features[l_idx].kernel_size
    (pad1, pad2) = net.features[l_idx].padding
    
    # Truncate all layers before l_idx and wrap the current conv layer as a PatchConvLayer class.
    (s1, s2) = net.features[l_idx].stride
    patchnet.features = net.features[l_idx:]
    patchnet.features[0] = layer

    count = -1
    
    #Todo: Directly get the weights from layer and avoid this loop. This is done in other verifiers like agop_fc
    for idx, p in enumerate(net.parameters()):
        
        # This logic of identifying layer_idx+1th parameters is not generic. It will fail if Batchnorm2d is present
        # Why not get weights directly from the layer object?
        if len(p.shape) > 1:
            count += 1
        if count == layer_idx:
            M = p.data #(k,c,q,s)
            _, ki, q, s = M.shape
            
            # Build W which is a (k, c*q*s) matrix
            M = M.reshape(-1, ki*q*s)
            
            # Compute WtW which is (c*q*s,c*q*s) matrix
            M = torch.einsum('nd, nD -> dD', M, M)
            
            # Build W0 from the untrained net which is a (k, c*q*s) matrix
            M0 = [p for p in init_net.parameters()][idx]          
            M0 = M0.reshape(-1, ki*q*s)
            
            # Compute W0tW0 which is (c*q*s,c*q*s) matrix
            M0 = torch.einsum('nd, nD -> dD', M0, M0)
            break

    return net, patchnet, M, M0, l_idx, [(q, s), (pad1,pad2), (s1,s2)]


def get_grads(net, patchnet, trainloader,
              kernel=(3,3), padding=(1,1),
              stride=(1,1), layer_idx=0):
    net.eval()
    net.cuda()
    patchnet.eval()
    patchnet.cuda()
    M = 0
    q, s = kernel
    pad1, pad2 = padding
    s1, s2 = stride

    # Num images for taking AGOP (Can be small for early layers)
    MAX_NUM_IMGS = 10

    for idx, batch in enumerate(trainloader):
        print("Computing GOP for sample " + str(idx) + \
              " out of " + str(MAX_NUM_IMGS))
        imgs, _ = batch
        with torch.no_grad():
            imgs = imgs.cuda()        
            # Run the first half of the network wrt to the current layer 
            imgs = net.features[:layer_idx](imgs).cpu() #(n,c,h,w)
        patches = patchify(imgs, (q, s), (s1,s2), padding=(pad1,pad2))#(n,h_out,w_out,c,q,s)
        patches = patches.cuda()
        #print(patches.shape)
        M += egop(patchnet, patches).cpu()
        del imgs, patches
        torch.cuda.empty_cache()
        if idx >= MAX_NUM_IMGS:
            break
    net.cpu()
    patchnet.cpu()
    return M


def min_max(M):
    return (M - M.min()) / (M.max() - M.min())


def correlation(A, B):
    M1 = A.clone()
    M2 = B.clone()
    M1 -= M1.mean()
    M2 -= M2.mean()

    norm1 = norm(M1.flatten())
    norm2 = norm(M2.flatten())

    return torch.sum(M1.cuda() * M2.cuda()) / (norm1 * norm2)


def verify_NFA(net, init_net, trainloader, layer_idx=0):


    net, patchnet, M, M0, l_idx, conv_vals = load_nn(net,
                                                     init_net,
                                                     layer_idx=layer_idx)
    (q, s), (pad1, pad2), (s1, s2) = conv_vals    

    G = get_grads(net, patchnet, trainloader,
                  kernel=(q, s),
                  padding=(pad1, pad2),
                  stride=(s1, s2),
                  layer_idx=l_idx)
    print("Shpae after gradients: ", G.shape)
    G = sqrt(G)
    Gop = G.clone()
    
    print("Correlation between Initial and Trained CNFM: ", correlation(M0, M))
    print("Correlation between Initial CNFM and Trained AGOP: ", correlation(M0, G))
    print("Correlation between Trained CNFM and Trained AGOP: ", correlation(M, G))
    return Gop 
    #return i_val.data.numpy(), r_val.data.numpy()

def vis_transform_image(net, img, G, layer_idx=0):

    count = -1
    
    # Computes WtW for the weights(ignoring its bias) of layer_idx+1 the conv layer
    for idx, p in enumerate(net.parameters()):
        if len(p.shape) > 1:
            count += 1
        if count == layer_idx:
            M = p.data
            print(M.shape)
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


def sqrt(G):
    U, s, Vt = svd(G)
    s = torch.pow(s, 1./2)
    G = U @ torch.diag(s) @ Vt
    return G


def main():

    # Adjust to index conv layers in VGGs
    idxs = list(range(8))

    fname = 'csv_logs/test.csv'
    outf = open(fname, 'w')

    net = models.vgg11(weights="DEFAULT")
    #init_net is used as a reference untrained network.
    init_net = models.vgg11(weights=None)

    # Modules is unused.
    modules= list(net.children())[:-1]
    modules += [nn.Flatten(), list(net.children())[-1]]

    # Set path to imagenet data
    path = None

    trainloader, _ = dataset.get_imagenet(batch_size=2, path=path)

    for idx in idxs:
        i_val, r_val = verify_NFA(net, init_net, trainloader, layer_idx=idx)
        print("Layer " + str(idx+1) + ',' + str(i_val) + ',' + str(r_val), file=outf, flush=True)


#if __name__ == "__main__":
    #main()