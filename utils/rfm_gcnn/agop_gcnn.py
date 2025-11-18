''' This module does the following
1. Scan the network for conv layers
2. For each gcnn conv layer compute W^TW of eq 3
3. For each gcnn conv layer compute the AGOP(AJOP in case of multiple outputs)
4. For each gcnn conv layer print the pearson correlation between 2 and 3
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.func import jacrev
#from torch.nn.functional import pad
from torch.linalg import norm, svd, eig
from torchvision import models
import random
import numpy as np
import visdom

from utils.rfm_gcnn.utils import patchify, trans_filter, sqrt, matrix_power_eigendecomposition, correlation, min_max, get_nfm
from utils.groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M
from groupy.gconv.make_gconv_indices import *
from copy import deepcopy

#Todo: Import these blocks directly by avoiding model2
#from trained_models.CIFAR.model2.model2 import BasicBlock, Bottleneck

SEED = 2323

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

#vis = visdom.Visdom('http://127.0.0.1', use_incoming_socket=False)
#vis.close(env='main')


class PatchConvLayer(nn.Module):
    def __init__(self, conv_layer):
        super().__init__()
        self.layer = conv_layer #(k,c,q,s)
        #inds = make_c4_z2_indices(self.layer.ksize)
       
    def forward(self, patches):
        if(len(patches.shape)==7):
            patches= patches[:,0,:,:,:,:,:]
        tw = trans_filter(self.layer.weight, self.layer.inds)
        tw_shape = (self.layer.out_channels * self.layer.output_stabilizer_size,
                    self.layer.in_channels * self.layer.input_stabilizer_size,
                    self.layer.ksize, self.layer.ksize)
        tw = tw.view(tw_shape)
        #print("tw shape",tw.shape)
        #print("Patch_shape", patches.shape)
        out = torch.einsum('nhwcqr, kcqr -> nhwk', patches, tw)
        n, w, h, k = out.shape
        out = out.transpose(1, 3).transpose(2, 3) #(n,k,h_out,w_out)
        out = out.view(n, self.layer.out_channels, self.layer.output_stabilizer_size, h, w)
        #print("out_shape", out.shape)
        return out


class PatchBasicBlock(nn.Module):

    def __init__(self, block_layer):
        super().__init__()
        self.layer = block_layer

    def forward(self, X):          
        #print(X.shape)
        x1 = X[:,0,:,:,:,:,:] #(1,w_out, h_out, c, q, s)
        x2 = X[:,1,:,:,:,:,:] #(1,w_out, h_out, c, q, s)
        o = self.layer.features(x1)
        if(self.layer.shortcut):
            z = self.layer.shortcut(x2)
            o+=z
        o = self.layer.lrelu(o)
        return o
        
    

def get_jacobian(net, data, c_idx=0, chunk=100):
    with torch.no_grad():
        def single_net(X):
            # x is (2, w_out,h_out,c,q,s)
            return net(X.unsqueeze(0))[:,c_idx*chunk:(c_idx+1)*chunk].squeeze(0)
        # Parallelize across the images.
        #data: (bs, 2, w_out, h_out, c, q, s)
        return torch.vmap(jacrev(single_net))(data) #(bs, chunk, 2, w_out, h_out, c, q, s)

def egop(model, z, classes=10, chunk_idxs=10):
    ajop = 0
    c = classes
    #Chunking is done to compute jacobian as sum of smaller size matrices using outer product. This saves memory
    chunk = c // chunk_idxs
    chunk_list = []
    for i in range(chunk_idxs):
        J = get_jacobian(model, z, c_idx=i, chunk=chunk) #(n, chunk, 2, w_out, h_out, c, q, s)
        J= J[:,:,0,:,:,:,:,:]
        n, c, w, h, _, _, _ = J.shape
        J = J.transpose(1, 3).transpose(1, 2) #(n, w_out, h_out, chunk, c, q, s)
        grads = J.reshape(n*w*h, c, -1) #(n*w_out*h_out, chunk, c*q*s)
        chunk_list.append(grads)
        #Clarify: Where is mean taken      
        #ajop += torch.einsum('ncd, ncD -> dD', grads, grads) #(c*q*s,c*q*s)
        #del J, grads
        #torch.cuda.empty_cache()
    return torch.cat(chunk_list, dim=1)
    return ajop


def load_nn(net, init_net, layer_idx=0):
    
    count = 0
    # Get the layer_idx+1 th conv layer
    for idx, m in enumerate(net.features):
        if isinstance(m, (P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M, BasicBlock, Bottleneck)):
            count += 1
        if count-1 == layer_idx:
            l_idx = idx
            break

    print("l_idx",l_idx)
    if(isinstance(net.features[l_idx],(P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M))):
        
        # Construct patchnet
        patchnet = deepcopy(net)
        temp = deepcopy(net.features[l_idx])
        conv_layer = PatchConvLayer(temp)
        
        #Truncate all layers before l_idx    
        patchnet.features = net.features[l_idx:]
        patchnet.features[0] = conv_layer
        
        #layer whose CNFM we need
        layer = deepcopy(net.features[l_idx])
        layer_init = deepcopy(init_net.features[l_idx])     
        
    else:   
        
        # Construct patchnet
        patchnet = deepcopy(net)
        temp_block = deepcopy(net.features[l_idx])
        conv_layer = PatchConvLayer(temp_block.features[0])
        temp_block.features[0]= conv_layer
        if(len(temp_block.shortcut)>0):
           short_layer = PatchConvLayer(temp_block.shortcut[0])
           temp_block.shortcut[0] = short_layer
        else:
           temp_block.shortcut= None
            
        temp_block = PatchBasicBlock(temp_block)
        
        #Truncate all layers before l_idx    
        patchnet.features = net.features[l_idx:]
        patchnet.features[0] = temp_block
        
        #layer whose CNFM we need
        layer = deepcopy(net.features[l_idx].features[0])
        layer_init = deepcopy(init_net.features[l_idx].features[0])        
                
    # Compute WtW which is (c*q*s,c*q*s) matrix
    M = get_rfm(layer)
    M0 = get_rfm(layer_init)

    return net, patchnet, M, M0, l_idx, [(q, s), (pad1,pad2), (s1,s2)], in_channels, input_stabilizer_size


def get_grads(net, in_channels, input_stabilizer_size, patchnet, trainloader,
              kernel=(3,3), padding=(1,1),
              stride=(1,1), layer_idx=0, max_batches=2, classes=10, chunk_size=10, centering=True):
    net.eval()
    net.cuda()
    patchnet.eval()
    patchnet.cuda()
    q, s = kernel
    pad1, pad2 = padding
    s1, s2 = stride

    M = 0
    ajop = 0
    J_sum=0
    n=0
    
    c = classes
    #Chunking is done to compute jacobian as sum of smaller size matrices using outer product. This saves memory
    chunk = c // chunk_size
    chunk_list = []
    # bs = len(list(trainloader)[0]) 
    bs=128

    for i in range(chunk_size):
        grads = []
        J_sum=0
        print("************* Chunk"+str(i)+"*************")
        for idx, batch in enumerate(trainloader):
            #print("Computing GOP for sample " + str(idx) + \
                  #" out of " + str(max_batches))
            imgs, _ = batch
            #imgs= imgs.double()
            with torch.no_grad():
                imgs = imgs.cuda()     
                imgs = imgs.float()
                # Run the first half of the network wrt to the current layer 
                imgs = net.features[:layer_idx](imgs).cpu() #(bs,c,h,w)
            patches = patchify(imgs, in_channels, input_stabilizer_size, 
                               (q, s), (s1,s2), padding=(pad1,pad2))#(bs,w_out,h_out,c,q,s)
            p_copy = deepcopy(patches)
            patches = patches.cuda()
            p_copy = p_copy.cuda()
            c_patches = torch.stack([patches, p_copy], dim=1) #(bs,2,w_out,h_out,c,q,s)
            J = get_jacobian(patchnet, c_patches, c_idx=i, chunk=chunk)
            J= J[:,:,0,:,:,:,:,:]
            bs, c, w, h, _, _, _ = J.shape
            print(J.shape)
            J = J.transpose(1, 3).transpose(1, 2) #(bs, w_out, h_out, chunk, c, q, s)
            J = J.reshape(bs*w*h, c, -1) #(bs*w_out*h_out, chunk, c*q*s)
            if centering:
                J_sum += torch.sum(J, dim=0).unsqueeze(0)  
            J= J.cpu()
            grads.append(J)          
            n += bs         
            #M += egop(patchnet, c_patches, classes, chunk_size).cpu()
            #Js.append(egop(patchnet, c_patches, classes, chunk_size).cpu()) 
            del imgs, patches, p_copy, c_patches, J
            torch.cuda.empty_cache()
            if idx >= max_batches:
                break
       
        if centering:
            J_mean = J_sum*1/(n*w*h) #(1, chunk, c*q*s)
            
        for batch_idx, J in enumerate(grads):
            J = J.cuda()
            if centering:
                J= J- J_mean
            M += torch.einsum('ncd,ncD->dD', J, J).cpu()
            del J
        torch.cuda.empty_cache()
        
    #net.cpu()
    #patchnet.cpu()
    return M*1/n


def verify_NFA(net, init_net, trainloader, layer_idx=0, max_batches=2, classes=10, chunk_size=10, alpha=0.5, centering= True):

    #net = net.double()
    #init_net = init_net.double()
    
    net, patchnet, M, M0, l_idx, conv_vals, in_channels, input_stabilizer_size = load_nn(net,
                                                     init_net,
                                                     layer_idx=layer_idx)
    (q, s), (pad1, pad2), (s1, s2) = conv_vals
  

    G = get_grads(net, in_channels, input_stabilizer_size, patchnet, trainloader,
                  kernel=(q, s),
                  padding=(pad1, pad2),
                  stride=(s1, s2),
                  layer_idx=l_idx, max_batches=max_batches, classes=classes, chunk_size=chunk_size, centering=centering)
    
    print("Shape after gradients: ", G.shape)
    #G = sqrt(G)
    G = matrix_power_eigendecomposition(G, alpha)
    Gop = G.clone()
    
    print("Correlation between Initial and Trained CNFM: ", correlation(M0, M))
    print("Correlation between Initial CNFM and Trained AGOP: ", correlation(M0, G))
    print("Correlation between Trained CNFM and Trained AGOP: ", correlation(M, G))

    del patchnet
    #print("Final: ", i_val, r_val)
    return (Gop, correlation(M, G))
    #return i_val.data.numpy(), r_val.data.numpy()








#TODO: ADD a visualizer for the image

#if __name__ == "__main__":
    #main()