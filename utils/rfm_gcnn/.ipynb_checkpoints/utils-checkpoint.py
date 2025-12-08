'''Helper functions.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.nn.functional import pad
from torch.linalg import norm, svd

import os
import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm, fractional_matrix_power
from matplotlib import pyplot as plt
import math
from copy import deepcopy
from einops import rearrange


#from groupy.gconv.make_gconv_indices import *

plt.switch_backend('Agg')

def float_x(data):
    '''Set data array precision.'''
    return np.float32(data)


################################ MATRIX HELPER FUNCTIONS ###############################################
'''
def matrix_power(M, power):
    """
    Compute the power of a matrix.
    :param M: Matrix to power.
    :param power: Power to raise the matrix to.
    :return: Matrix raised to the power - M^{power}.
    """
    if len(M.shape) == 2:
        assert M.shape[0] == M.shape[1], "Matrix must be square"
        M_cpu = M.cpu()
        original_device = M.device
        try:
            # gpu square root
            S, U = torch.linalg.eigh(M)
            S[S<0] = 0.
            return U @ torch.diag(S**power) @ U.T
        except:
            # stable cpu square root
            M_cpu.diagonal().add_(1e-8)
            if power == 0.5:
                sqrtM = sqrtm(M_cpu)
            else:
                sqrtM = fractional_matrix_power(M_cpu, power)
            sqrtM = torch.from_numpy(sqrtM).to(original_device)
            return sqrtM
    elif len(M.shape) == 1:
        assert M.shape[0] > 0, "Vector must be non-empty"
        M[M<0] = 0.
        return M**power
    else:
        raise ValueError(f"Invalid matrix shape for square root: {M.shape}")

'''

def matrix_power_eigendecomposition(G, alpha):   
    #TODO: Check if fractional_matrix_power from scipy can be used
    eigenvalues, eigenvectors = torch.linalg.eigh(G)    
    # Raise the eigenvalues to the power alpha
    # Clamp to zero to handle potential small negative eigenvalues due to
    # floating-point inaccuracies, which would result in NaN for fractional powers.
    powered_eigenvalues = torch.clamp_min(eigenvalues, 0).pow(alpha)
    V = eigenvectors
    Lambda_powered = torch.diag(powered_eigenvalues)
    powered_matrix = V @ Lambda_powered @ V.T
    
    return powered_matrix


def sqrt(G):
    #TODO: Check if sqrtm from scipy can be used
    U, s, Vt = svd(G)
    s = torch.pow(s, 1./2)
    G = U @ torch.diag(s) @ Vt
    return G

def correlation(M, G):
    A = M.clone()
    B = G.clone()
    A -= A.mean()
    B -= B.mean()
    A = A.double()
    B = B.double()
    normM = norm(A.flatten())
    normG = norm(B.flatten())

    corr = torch.dot(A.flatten(), B.flatten()) / (normM * normG)
    return corr
    
def cosine_similarity(vector_a, vector_b):
  
    # Calculate the dot product
    dot_product = torch.dot(vector_a, vector_b)

    # Calculate the Euclidean (L2) norm (magnitude) of each vector
    norm_a = torch.linalg.norm(vector_a)
    norm_b = torch.linalg.norm(vector_b)

    # Handle cases where one or both vectors are zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0 # Cosine similarity is undefined or typically set to 0 for zero vectors

    # Compute cosine similarity
    cosine_sim = dot_product / (norm_a * norm_b)

    return cosine_sim

def min_max(M):
    return (M - M.min()) / (M.max() - M.min())


################################ IMAGE HELPER FUNCTIONS ###############################################

def patchify(x, layer, pad_type='zeros'):
    '''
        Given an input image (bs,c,h,w) generate (bs,h_out,w_out,c,q,s) respecting stride,padding, 
        w_out is number of pathces along the width for the given stride after padding
        h_out is number of pathces along the height for the given stride after padding
        (q,s) is the kernel dimensions 
    '''
    #TODO: Compare with the padding done in gcnn. Ensure they are same.
    #TODO: double check width height order
    
    input_shape = x.size()
    in_channels = layer.in_channels
    ip_stab = layer.input_stabilizer_size
    patch_size = layer.kernel_size 
    stride_size = layer.stride
    padding = layer.padding 
    
    x = x.reshape(input_shape[0], in_channels*ip_stab, input_shape[-2], input_shape[-1])
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
        
    patches = x.unfold(2, q1, s1).unfold(3, q2, s2) #(bs, c, h_out, w_out, q, s)
    #print("Image Shape1",patches.shape)
    patches = patches.transpose(1, 3).transpose(1, 2) #(bs, h_out, w_out, c, q, s) 
    #print("Image Shape2",patches.shape)
    return patches

'''
def expand_image(X, ps=3, pad_mode="circular"):
    """
    X : (n, c, p, q)
    out : (n, c, p*ps, q*ps)
    """

    n, c, p, q = X.shape

    pad_sz = ps//2
    if pad_mode=="zero":
        pad = (pad_sz,pad_sz,pad_sz,pad_sz)
        X_patched = F.pad(X, pad)
    elif pad_mode=="circular":
        X_patched = torch.from_numpy(np.pad(X, ((0,0),(0,0),(pad_sz,pad_sz),(pad_sz,pad_sz)), mode='wrap'))

    X_patched = X_patched.unfold(2,ps,1).unfold(3,ps,1) # (n, c, p, q, ps, ps)
    X_patched = X_patched.transpose(-2,-3) # (n, c, p, ps, q, ps)
    X_expanded = X_patched.reshape(n,c,p*ps,q*ps)
    return X_expanded
'''

def reduce_image(X, depth, ps=3):
    """
    X : (n, c, p*ps, q*ps)
    out : (n, c, p, q)
    """
    n, c, P, Q = X.shape
    p = P//ps
    q = Q//ps

    X = X.reshape(n, c, p, ps, q, ps)
    if depth == 0:
        return X.norm(dim=(3,5))
    else:
        X = X.norm(dim=(3,5))
        #X = torch.max(X, dim=1)[0]
        return X
        #X = X**2 
        #X = X.sum(dim=(1,3,5))
        #return X.sqrt()

    #X = torch.permute(X, (0, 1, 3, 5, 2, 4))
    #X = X.reshape(n, c*ps*ps, p*q)
    #pad_sz = ps//2
    #folded = fold(X, output_size=(p, q), kernel_size=(ps, ps), padding=(pad_sz, pad_sz))

    #ones = torch.ones(X.shape)
    #ones = fold(ones, output_size=(p, q), kernel_size=(ps, ps), padding=(pad_sz, pad_sz))
    #return folded/ones



def multiply_patches(X, M, ps=3):
    """
    Applies the covariance matrix M
    X : (n, c, p*ps, q*ps)
    M_ : (c*w*h, c*w*h)
    out : (n, c, p*ps, q*ps)
    """
    n = X.shape[0]
    chunk = 5000
    leftover_bool = int(n%chunk>0)
    batches = np.array_split(np.arange(n), n//chunk + leftover_bool)

    M = sqrt(M)

    Xs = []
    for i, b in enumerate(batches):
        Xb = X[b]
        m, c, P, Q = Xb.shape
        p = P//ps
        q = Q//ps
        Xb = rearrange(Xb, 'm c (p h) (q w) -> (m p q) (c h w)', p=p, q=q, w=ps, h=ps)
        #TODO: Check the datatypes of Xb and M.
        Xb = Xb @ M
        #Xb = rearrange(Xb, '(m p q) (c w h) -> m c (p w) (q h)', m=m, p=p, q=q, c=c, w=ps, h=ps)
        Xb = rearrange(Xb, '(m p q) (c h w) -> m (c h w) p q', m=m, p=p, q=q, c=c, w=ps, h=ps) 
        Xb = norm(Xb, dim=1, keepdim=True)
        Xs.append(Xb)
    return torch.cat(Xs, dim=0)



def vis(tensor_input, pwd, fname, title="NFM vs RFM"):
    """
    Visualizes an N x P x 2 x C x H x W tensor, averaging across channels (C)
    and plotting 'P' poses with precisely centered and slightly lowered titles.
    """
    # 1. Data Preprocessing
    if torch.is_tensor(tensor_input):
        data = tensor_input.detach().cpu().numpy() 
    else:
        data = tensor_input

    N, P, _, C_original, H, W = data.shape
    num_poses = P
    
    # *** CHANNEL AVERAGING ***
    data_averaged = np.mean(data, axis=3, keepdims=True)
    
    # 2. Setup Plotting Parameters
    rows = N
    cols = num_poses * 2 
    
    save_path = f'{pwd}/images'
    os.makedirs(save_path, exist_ok=True)

    COLOR_MAP = 'hot' 
    
    # Adjusted strip width
    FIG_SIZE_X = cols * 2.5  
    FIG_SIZE_Y = rows * 3.5 
    
    fig, axes = plt.subplots(rows, cols, figsize=(FIG_SIZE_X, FIG_SIZE_Y), squeeze=False) 

    # Define common scale
    min_val = data_averaged.min()
    max_val = data_averaged.max()
    
    conditions = [
        ("backprob", 0),
        ("No backprob ", 1)
            ]

    # 3. Plotting Loop (MUST run before centering to establish axis positions)
    for n in range(N):
        row_title = f"Sample {n+1}"
        axes[n, 0].set_ylabel(row_title, rotation=90, labelpad=15, fontsize=12, weight='bold')

        for p in range(num_poses):
            for cond_idx, (cond_title, data_index) in enumerate(conditions): 
                col_idx = (p * 2) + cond_idx
                ax = axes[n, col_idx]
                
                composite_image = data_averaged[n, p, data_index, 0]
                ax.imshow(composite_image, cmap=COLOR_MAP, vmin=min_val, vmax=max_val)
                
                # Set the condition title (With/Without backprop) only for the top row (n=0)
                if n == 0:
                    short_title = cond_title.split('-')[0] 
                    ax.set_title(short_title, fontsize=10) 
                
                ax.axis('off')

    # Apply tight_layout BEFORE calculating text positions for accurate coordinates
    # We must ensure there is enough space left at the top for the titles
    plt.tight_layout(rect=[0.02, 0.01, 1, 0.95]) 
    
    fig.suptitle(f"{title} (Averaged across {C_original} Channels)", fontsize=16, y=0.99)

    y_pos = 0.94 
    
    for p in range(num_poses):
        start_col = p * 2
        end_col = p * 2 + 1
        
        # Get the axes objects for the two columns in the TOP row (n=0)
        ax_start = axes[0, start_col]
        ax_end = axes[0, end_col]
        
        # Get the bounding box of the two axes in figure coordinates
        left_fig = ax_start.get_position().extents[0] 
        right_fig = ax_end.get_position().extents[2] 
        
        # Calculate the exact center x-position
        center_x_fig = (left_fig + right_fig) / 2
        
        # Place the Pose title using the calculated center_x_fig and new y_pos
        fig.text(center_x_fig, y_pos, f"Pose {p+1}", 
                 ha='center', fontsize=14, weight='bold', 
                 transform=fig.transFigure)

    full_file_path = f'{save_path}/{fname}'
    plt.savefig(full_file_path)

    print(f"Visualization saved to {full_file_path}")



def nfmvrfm(img, layer, M, pose=0):
    #TODO: Handle other poses
    pat = patchify(img, layer)
    pat = pat.transpose(2,3).transpose(1,2).transpose(3,4) #(bs, c, h_out, q, w_out, s)
    pat = pat.reshape(pat.shape[0], pat.shape[1], pat.shape[2]*pat.shape[3], pat.shape[4]*pat.shape[5])

    N =  get_nfm(layer, pose)
    o1 = multiply_patches(pat.cuda(), N.cuda(), ps=layer.ksize)
    #o1 = reduce_image(o1, 0, ps=layer.ksize)

    tup = find_covariance_matrix_m(layer, M, pose=pose)
    M = torch.from_numpy(tup[0])
    M = M.float()
    #M = ut.sample_normal(M, layer.weight.shape[0])
    o2 = multiply_patches(pat.cuda(), M.cuda(), ps=layer.ksize)
    #o2 = reduce_image(o2, 0, ps=layer.ksize)
    combined_tensor = torch.stack((o1, o2), dim=1)
    return combined_tensor
    

################################ GCNN HELPER FUNCTIONS ###############################################

def trans_filter(w, inds):
    
    #TODO: Reference this function directly from the SplitConv2d class of gcnn.
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64)
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
    w_indexed = w_indexed.reshape(w_indexed.size()[0], w_indexed.size()[1],
                                    inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3])
    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
    return w_transformed.contiguous()

def get_nfm(layer, pose=None):
    
    # Extract all the meta info of the current conv layer.
    (q, s) = layer.kernel_size
    (pad1, pad2) = layer.padding
    (s1, s2) = layer.stride 
    in_channels = layer.in_channels
    input_stabilizer_size = layer.input_stabilizer_size
    tw = trans_filter(layer.weight, layer.inds)
    
    
    if pose is not None:      
        tw = tw[:, pose, :, :, :, :]
        tw_shape = (layer.out_channels, layer.in_channels * layer.input_stabilizer_size,
                            layer.ksize, layer.ksize)
        W = tw.reshape(tw_shape)
    else:     
        tw = trans_filter(layer.weight, layer.inds)   
        tw_shape = (layer.out_channels * layer.output_stabilizer_size,
                            layer.in_channels * layer.input_stabilizer_size,
                            layer.ksize, layer.ksize)
        W = tw.reshape(tw_shape)
        
    k, ki, q, s= W.shape
    
    # Reshape W which into a (k, c*q*s) matrix. 
    W = W.reshape(-1, ki*q*s)
                
    # Compute WtW which is (c*q*s,c*q*s) matrix
    M = torch.einsum('nd, nD -> dD', W, W)
    return M

def get_permutation_matrix(v1, v2):
    
    if v1.shape != v2.shape or v1.ndim != 1:
        raise ValueError("v1 and v2 must be 1D tensors of the same shape.")

    n = v1.size(0)
    perm_indices = torch.empty_like(v2, dtype=torch.long)
    for i in range(n):
        perm_indices[i] = (v1 == v2[i]).nonzero(as_tuple=True)[0][0]
   
    P = F.one_hot(perm_indices.long(), num_classes=n).float()

    return P

def sample_normal(covariance_matrix_M, num_rows_k):
   
    # Create a mean vector of zeros, compatible with the dimensions of M
    device = covariance_matrix_M.device
    dtype = covariance_matrix_M.dtype
    matrix_dim_t = covariance_matrix_M.shape[0]
    #mean_vector = torch.zeros(matrix_dim_t, dtype=dtype)
    mean_vector = torch.zeros((matrix_dim_t,), dtype=dtype, device=device)
    #mean_vector.to(device)
    try:
        mvn = distributions.MultivariateNormal(loc=mean_vector, covariance_matrix=covariance_matrix_M)
    except Exception as e:
        print(f"Error creating MultivariateNormal distribution: {e}")
        print("Please ensure your covariance_matrix_M is symmetric positive definite.")
        # Attempt to make it PSD for robustness in example, by adding a small diagonal perturbation
        # In a real scenario, you'd ensure your input M is correctly formed.
        min_eigval = torch.min(torch.linalg.eigvalsh(covariance_matrix_M)).item()
        if min_eigval <= 0:
            print(f"Warning: Covariance matrix has non-positive eigenvalue ({min_eigval:.2e}). Adding jitter.")
            covariance_matrix_M = covariance_matrix_M + torch.eye(matrix_dim_t, device=device, dtype=dtype) * (abs(min_eigval) + 1e-6)
            mvn = distributions.MultivariateNormal(loc=mean_vector, covariance_matrix=covariance_matrix_M)


    # Sample num_rows_k times from the distribution
    # The .sample() method will return a tensor of shape (num_rows_k, matrix_dim_t)
    sampled_matrix = mvn.sample((num_rows_k,))

    return sampled_matrix

def find_covariance_matrix_m(layer, S_target_np, pose=0, solver_name='SCS'):

    dummy= deepcopy(layer)
    dummy.out_channels = 1
    temp = torch.arange(dummy.in_channels*dummy.input_stabilizer_size*dummy.ksize*dummy.ksize, dtype=torch.float)
    temp= temp.reshape(dummy.out_channels,dummy.in_channels,dummy.input_stabilizer_size,dummy.ksize,dummy.ksize)
    dummy.weight = nn.Parameter(temp)
    dummy.inds = dummy.make_transformation_indices()
    tw = trans_filter(dummy.weight, dummy.inds)
    tw_shape = (dummy.out_channels * dummy.output_stabilizer_size,
                        dummy.in_channels * dummy.input_stabilizer_size,
                        dummy.ksize, dummy.ksize)
    tw = tw.reshape(tw_shape)
    tw_test= tw.reshape(dummy.out_channels, dummy.output_stabilizer_size,
                        dummy.in_channels , dummy.input_stabilizer_size,
                        dummy.ksize, dummy.ksize)
    
    print("tw_shape",tw_test.shape)

    P_matrices = []
    v1 = tw_test[0][pose].reshape(-1) 
    for i in range(tw_test.shape[1]):
        v2 = tw_test[0][i].reshape(-1) 
        P = get_permutation_matrix(v1, v2)
        print(P.shape)
        P_matrices.append(P)
        # Verification: Check that P @ v1 gives v2
        v1_col = v1.unsqueeze(1)
        v2_calc = P @ v1_col
        v2_calc= v2_calc.squeeze(1)
        
        print("\nVerification (P_corrected @ v1):\n", v2_calc)
        print("\nAre they equal? (within tolerance):", torch.allclose(v2_calc, v2))
        
    M_dim = S_target_np.shape[0]
    M = cp.Variable((M_dim, M_dim), symmetric=True)
    sum_term = 0
    for P in P_matrices:
        sum_term += P @ M @ P.T

    objective = cp.Minimize(cp.norm(sum_term - S_target_np, 'fro'))
    
    constraints = [M >> 0]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=solver_name)
    except Exception as e:
        print(f"An unexpected error occurred during solving: {e}")
        return None, None, "Error"
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return M.value, problem.value, problem.status
    else:
        print(f"Problem did not solve to optimality. Status: {problem.status}")
        print("This could mean the problem is infeasible, unbounded, or the solver failed to converge.")
        return None, problem.value, problem.status


################################ LOADER HELPER FUNCTIONS ###############################################

def get_data_from_loader(data_loader, layer, num_classes):
    """
    Get data from a data loader.
    :param data_loader: Torch DataLoader to get data from.
    :return: Tuple of tensors - (X, y).
    """
    X, y = [], []
    
    for idx, batch in enumerate(data_loader):
        inputs, labels = batch
        inputs = patchify(inputs, layer) 
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], inputs.shape[2], -1)
        X.append(inputs)
        y.append(F.one_hot(labels, num_classes).to(torch.float32))
    return torch.cat(X, dim=0), torch.cat(y, dim=0)
