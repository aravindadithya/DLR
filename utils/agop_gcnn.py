''' This module does the following
1. Scan the network for conv layers
2. For each gcnn conv layer compute W^TW of eq 3
3. For each gcnn conv layer compute the AGOP(AJOP in case of multiple outputs)
4. For each gcnn conv layer print the pearson correlation between 2 and 3
'''

import torch
import torch.nn as nn
import random
import numpy as np
from torch.func import jacrev
from torch.nn.functional import pad
from torch.linalg import norm, svd
from torchvision import models
import visdom
from torch.linalg import norm, eig
import random
import torch.backends.cudnn as cudnn
from torch.linalg import norm
from torchvision import models
import torch.nn.functional as F
from utils.groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M
#Todo: Import these blocks directly by avoiding model2
from trained_models.CIFAR.model2.model2 import BasicBlock, Bottleneck
from groupy.gconv.make_gconv_indices import *
from copy import deepcopy

SEED = 2323

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

vis = visdom.Visdom('http://127.0.0.1', use_incoming_socket=False)
vis.close(env='main')

def patchify(x, in_channels, ip_stab, patch_size, stride_size, padding=None, pad_type='zeros'):
    '''
        Given an input image (bs,c,h,w) generate (bs,w_out,h_out,c,q,s) respecting stride,padding, 
        w_out is number of pathces along the width for the given stride after padding
        h_out is number of pathces along the height for the given stride after padding
        (q,s) is the kernel dimensions 
    '''
    input_shape = x.size()
    #TODO: The last two shapes look swapped. This was the same order in cohens code too. For square ips 
    #there is no effect. However for rect ips what would happen?
    x = x.view(input_shape[0], in_channels*ip_stab, input_shape[-2], input_shape[-1])
    #x = x.view(input_shape[0], in_channels*ip_stab, input_shape[-2], input_shape[-1])
    q1, q2 = patch_size
    s1, s2 = stride_size
    #print("Image Shape",x.shape)
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
        
    patches = x.unfold(2, q1, s1).unfold(3, q2, s2) #(bs, c, h_out, w_out, q, s)
    #print("Image Shape1",patches.shape)
    patches = patches.transpose(1, 3).transpose(1, 2) #(bs,w_out,h_out,c,q,s) 
    #print("Image Shape2",patches.shape)
    return patches

def trans_filter(w, inds):
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64)
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
    w_indexed = w_indexed.view(w_indexed.size()[0], w_indexed.size()[1],
                                    inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3])
    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
    return w_transformed.contiguous()
    
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
    
    # Extract all the meta info of the current conv layer.
    (q, s) = layer.kernel_size
    (pad1, pad2) = layer.padding
    (s1, s2) = layer.stride 
    in_channels = layer.in_channels
    input_stabilizer_size = layer.input_stabilizer_size
    
    # Extract W matrix
    tw = trans_filter(layer.weight, layer.inds)
    tw_shape = (layer.out_channels * layer.output_stabilizer_size,
                        layer.in_channels * layer.input_stabilizer_size,
                        layer.ksize, layer.ksize)
    M = tw.view(tw_shape)
    
    tw= trans_filter(layer_init.weight, layer_init.inds)
    tw_shape = (layer_init.out_channels * layer_init.output_stabilizer_size,
                        layer_init.in_channels * layer_init.input_stabilizer_size,
                        layer_init.ksize, layer_init.ksize)
    M0 = tw.view(tw_shape)
    
    k, ki, q,s= M.shape
                
    # Build W which is a (k, c*q*s) matrix. What to do with ip_stab
    M = M.reshape(-1, ki*q*s)
                
    # Compute WtW which is (c*q*s,c*q*s) matrix
    M = torch.einsum('nd, nD -> dD', M, M)

    k, ki, q,s= M0.shape

    # Build W which is a (k, c*q*s) matrix. What to do with ip_stab
    M0 = M0.reshape(-1, ki*q*s)

    # Compute WtW which is (c*q*s,c*q*s) matrix
    M0 = torch.einsum('nd, nD -> dD', M0, M0)

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



def sqrt(G):
    U, s, Vt = svd(G)
    s = torch.pow(s, 1./2)
    G = U @ torch.diag(s) @ Vt
    return G

def matrix_power_eigendecomposition(G, alpha):      
    eigenvalues, eigenvectors = torch.linalg.eigh(G)    
    # Raise the eigenvalues to the power alpha
    # Clamp to zero to handle potential small negative eigenvalues due to
    # floating-point inaccuracies, which would result in NaN for fractional powers.
    powered_eigenvalues = torch.clamp_min(eigenvalues, 0).pow(alpha)
    V = eigenvectors
    Lambda_powered = torch.diag(powered_eigenvalues)
    powered_matrix = V @ Lambda_powered @ V.T
    
    return powered_matrix


#TODO: ADD a visualizer for the image

#if __name__ == "__main__":
    #main()