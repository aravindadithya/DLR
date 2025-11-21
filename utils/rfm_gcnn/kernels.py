'''Implementation of kernel functions.'''

import torch
import numpy as np
from typing import Union



def euclidean_distances_M(samples, centers, M=None, squared=True):
    '''
    Computes the squared (Mahalanobis-like or Euclidean) distance for K features, 
    resulting in a matrix of shape (N, M, P, Q). Uses the expanded distance formula 
    to avoid creating the large difference tensor (N, M, P, Q, K).

    Inputs:
        samples: (N, P, Q, K)
        centers: (M, P, Q, K)
        M: (K, K) for Mahalanobis-like distance, or None for Euclidean.
    
    Returns:
        Tensor of shape (N, M) containing the summed squared distance across P and Q.
    '''
    
    # 1. Expand dimensions for necessary broadcasting: (N, 1, P, Q, K) and (1, M, P, Q, K)
    samples_expanded = samples.unsqueeze(1) # (N, 1, P, Q, K)
    centers_expanded = centers.unsqueeze(0) # (1, M, P, Q, K)

    if M is None:
        # --- Euclidean Distance: ||a||^2 + ||b||^2 - 2 a.T b ---
        
        # Norm Terms: ||a||^2 and ||b||^2
        samples_norm2_exp = samples_expanded.pow(2).sum(dim=-1, keepdim=True)  # (N, 1, P, Q, 1)
        centers_norm2_exp = centers_expanded.pow(2).sum(dim=-1, keepdim=True)  # (1, M, P, Q, 1)
        
        # Cross Term: 2 * a.T b
        dot_product = torch.sum(samples_expanded * centers_expanded, dim=-1, keepdim=True) # (N, M, P, Q, 1)
        
        # Squared Distance for corresponding patches (P, Q) only
        squared_dist_PQ_K = samples_norm2_exp + centers_norm2_exp - 2.0 * dot_product 
        
    else:
        # --- Mahalanobis Distance: a^T M a + b^T M b - 2 a^T M b ---
        
        # Step A: Compute Weighted Norm Terms (a^T M a and b^T M b)
        # 1. Apply M to each vector: a @ M -> (N, 1, P, Q, K)
        samples_M = torch.matmul(samples_expanded, M)
        centers_M = torch.matmul(centers_expanded, M)
        
        # 2. Compute Weighted Norm: (a @ M) * a -> sum over K -> (N, 1, P, Q, 1)
        samples_norm2_M = torch.sum(samples_M * samples_expanded, dim=-1, keepdim=True)
        centers_norm2_M = torch.sum(centers_M * centers_expanded, dim=-1, keepdim=True)
        
        # 1. Compute Cross Term: (a @ M) @ b.T
        # This is equivalent to summing the element-wise product of (a @ M) and b over K.
        # Shape: (N, M, P, Q, K)
        cross_product_M = torch.sum(samples_M * centers_expanded, dim=-1, keepdim=True)
        
        # Step C: Combine Terms
        squared_dist_PQ_K = samples_norm2_M + centers_norm2_M - 2.0 * cross_product_M

    # Final clamping and sqrt for non-squared distance (if requested)
    if not squared:
        distances.clamp_(min=0).sqrt_()
        
    # Remove the singleton K dimension (which is 1)
    squared_dist_PQ = squared_dist_PQ_K.squeeze(-1) # Shape: (N, M, P, Q)

    # 2. Sum across P and Q patches
    distances = torch.sum(squared_dist_PQ, dim=(2, 3)) # Final shape: (N, M)
    
    return distances



'''
def euclidean_distances(samples, centers, squared=True):
    samples_norm2 = samples.pow(2).sum(-1)
    if samples is centers:
        centers_norm2 = samples_norm2
    else:
        centers_norm2 = centers.pow(2).sum(-1)

    distances = -2 * samples @ centers.T
    distances.add_(samples_norm2.view(-1, 1))
    distances.add_(centers_norm2)
    if not squared:
        distances.clamp_(min=0).sqrt_()

    return distances

def euclidean_distances_M(samples, centers, M, squared=True):
    if len(M.shape)==1:
        return euclidean_distances_M_diag(samples, centers, M, squared=squared)

    samples_norm2 = ((samples @ M) * samples).sum(-1)

    if samples is centers:
        centers_norm2 = samples_norm2
    else:
        centers_norm2 = ((centers @ M) * centers).sum(-1)

    distances = -2 * (samples @ M) @ centers.T
    distances.add_(samples_norm2.view(-1, 1))
    distances.add_(centers_norm2)

    if not squared:
        distances.clamp_(min=0).sqrt_()

    return distances
'''


'''
def euclidean_distances_M_diag(samples, centers, M, squared=True):
    "assumes M is a diagonal matrix"
    samples_norm2 = ((samples * M) * samples).sum(-1)

    if samples is centers:
        centers_norm2 = samples_norm2
    else:
        centers_norm2 = ((centers * M) * centers).sum(-1)

    distances = -2 * (samples * M) @ centers.T
    distances.add_(samples_norm2.view(-1, 1))
    distances.add_(centers_norm2)

    if not squared:
        distances.clamp_(min=0).sqrt_()

    return distances
'''

def laplacian(samples, centers, bandwidth):
    '''Laplacian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def laplacian_M(samples, centers, M, bandwidth):
    assert bandwidth > 0
    if M is None:
        kernel_mat = euclidean_distances(samples, centers, squared=False)
    else:
        kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def gaussian(samples, centers, bandwidth):
    '''Gaussian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=True)
    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

'''
def gaussian_M(samples, centers, M, bandwidth):
    assert bandwidth > 0
    if M is None:
        kernel_mat = euclidean_distances_M(samples, centers, squared=True)
    else:
        kernel_mat = euclidean_distances_M(samples, centers, M, squared=True)
    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat
'''

def gaussian_M(samples, centers, M, bandwidth):
    
    '''
    Computes the final summed Gaussian kernel result by ITERATING over 
    the P and Q patches, guaranteeing minimal memory usage.
    
    Returns: Final kernel matrix of shape (N, M).
    '''
    samples = samples.cuda()
    centres = centers.cuda()
    M = M.cuda()
    N, P, Q, K = samples.shape
    M_val, _, _, _ = centers.shape
    device = samples.device
    dtype = samples.dtype
    
    # Initialize the final result on the correct device/dtype
    final_kernel_mat = torch.zeros((N, M_val), device=device, dtype=dtype)
    
    # 1. Pre-calculate M-weighted centers for Mahalanobis
    if M is not None:
        # Pre-multiply all centers by M: (M*P*Q, K) -> (M*P*Q, K)
        centers_flat = centers.reshape(-1, K)
        centers_M_flat = centers_flat @ M
        centers_M = centers_M_flat.reshape(M_val, P, Q, K)

    # 2. Loop over P and Q patches (Guaranteed memory constraint)
    for p in range(P):
        for q in range(Q):
            # Extract current patches for samples (N, K) and centers (M, K)
            a = samples[:, p, q, :] # (N, K)
            b = centers[:, p, q, :] # (M, K)
            
            # --- Norm Terms Calculation (a^T M a and b^T M b) ---
            if M is None:
                # Euclidean Norm: ||a||^2 (N), ||b||^2 (M)
                a_norm2 = a.pow(2).sum(-1) # (N)
                b_norm2 = b.pow(2).sum(-1) # (M)
                
                # Cross Term: 2 * a.T b
                cross_term = 2.0 * (a @ b.T) # (N, M)
                
            else:
                # Mahalanobis Norm: a^T M a = (a @ M) * a
                # Reusing the pre-multiplied centers_M
                a_M = a @ M          # (N, K)
                b_M = centers_M[:, p, q, :] # (M, K)

                a_norm2 = (a_M * a).sum(-1) # (N)
                b_norm2 = (b_M * b).sum(-1) # (M)
                
                # Cross Term: 2 * a.T M b = 2 * (a @ M) @ b.T
                cross_term = 2.0 * (a_M @ b.T) # (N, M)
            
            # --- Kernel Calculation ---
            # Squared Distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.T b
            # Broadcasting: (N) -> (N, 1), (M) -> (1, M). Result: (N, M)
            squared_dist = a_norm2.unsqueeze(1) + b_norm2.unsqueeze(0) - cross_term 
            
            # Apply Gaussian Kernel
            # K_pq = exp(-gamma * D^2_pq)
            gamma = 1.0 / (2.0 * bandwidth ** 2)
            
            kernel_pq = squared_dist.clamp(min=0).mul(-gamma).exp()
            
            # Sum the kernel values
            final_kernel_mat.add_(kernel_pq)
            
    return final_kernel_mat



def dispersal(samples, centers, bandwidth, gamma):
    '''Dispersal kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.
        gamma: dispersal factor.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers)
    kernel_mat.pow_(gamma / 2.)
    kernel_mat.mul_(-1. / bandwidth)
    kernel_mat.exp_()
    return kernel_mat


#### NTK FUNCTIONS ####

def ntk_kernel(pair1, pair2):

    out = pair1 @ pair2.transpose(1, 0) + 1
    N1 = torch.sum(torch.pow(pair1, 2), dim=-1).view(-1, 1) + 1
    N2 = torch.sum(torch.pow(pair2, 2), dim=-1).view(-1, 1) + 1

    XX = torch.sqrt(N1 @ N2.transpose(1, 0))
    out = out / XX

    out = torch.clamp(out, -1, 1)

    first = 1/np.pi * (out * (np.pi - torch.acos(out)) \
                       + torch.sqrt(1. - torch.pow(out, 2))) * XX
    sec = 1/np.pi * out * (np.pi - torch.acos(out)) * XX
    out = first + sec

    # Set C below as small as possible for fast convergence
    # C = 1 on real data usually works well
    # set C > 1 if EigenPro is not converging
    C = 1
    return out / C


#### LAPLACIAN GEN FUNCTIONS #### 

def laplacian_gen(X: torch.Tensor, Z: torch.Tensor, sqrtM: torch.Tensor = None, L: float = 10.0, v: float = 1.0, diag: bool = False) -> torch.Tensor:
    """
    Optimized memory-efficient implementation of exponential kernel using batched tensor operations.
    
    Args:
        X: Input tensor of shape (n, d)
        Z: Input tensor of shape (m, d)
        sqrtM: Optional transformation matrix
        L: Length scale parameter (default: 10.0)
        exponent: Power parameter for the kernel (default: 1.0)
        batch_size: Number of dimensions to process at once (default: 50)
    
    Returns:
        Kernel matrix of shape (n, m)
    """
    n, d = X.shape
    m, d2 = Z.shape
    assert d == d2, "Feature dimensions must match"

    if sqrtM is not None:
        if diag:
            assert sqrtM.shape == (d,), "sqrtM must be a vector of length d"
            X = X * sqrtM.view(1, -1)
            Z = Z * sqrtM.view(1, -1)
        else:
            X = X @ sqrtM
            Z = Z @ sqrtM

    pdists = torch.cdist(X/L, Z/L, p=v) ## ||X/L-Z/L||_p = (\sum_{i=1}^d (|xi-zi|/L)^p)^(1/p)
    pdists_p = pdists**v ## \sum_{i=1}^d (|xi-zi|/L)^p
    return torch.exp(-1*pdists_p) ## \prod_{i=1}^d exp(-(|xi-zi|/L)^p)

def get_laplacian_gen_grad(
    x: torch.Tensor, 
    z: torch.Tensor, 
    sqrtM: Union[torch.Tensor, None], 
    v: float, 
    L: float,
    alphas: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes dk/dx for the kernel k(Mx, z) = ∏ exp(-|(Mx)_i - z_i|^v)
    
    Args:
        x: Input tensor (n, d_in) or (d_in,)
        z: Input tensor (m, d_out) or (d_out,)
        sqrtM: Transformation matrix (d_in, d_out)
        v: Exponent parameter
        L: bandwidth
        eps: Numerical stability term
        
    Returns:
        Gradient tensor of shape:
        - (n, m, d_in) if x is 2D and z is 2D
        - (m, d_in) if x is 1D and z is 2D
        - (d_in,) if both are 1D
    """
    # Ensure 2D tensors
    x = x.unsqueeze(0) if x.dim() == 1 else x
    z = z.unsqueeze(0) if z.dim() == 1 else z
    
    if sqrtM is None:
        sqrtM = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    
    # Transform x through linear layer
    Mx = x @ sqrtM / L
    z = z @ sqrtM / L
    
    pdists = torch.cdist(Mx, z, p=v) ## ||X/L-Z/L||_p = (\sum_{i=1}^d (|xi-zi|/L)^p)^(1/p)
    pdists_p = pdists**v ## \sum_{i=1}^d (|xi-zi|/L)^p
    k = torch.exp(-1*pdists_p) ## \prod_{i=1}^d exp(-(|xi-zi|/L)^p)
    
    # Compute gradient components for ∂k/∂(Mx)
    zero_mask = (pdists < eps) # (n, m)
    
    diff = x.unsqueeze(1) - z.unsqueeze(0)
    zero_mask_expanded = zero_mask.unsqueeze(-1)
    safe_abs = torch.where(zero_mask_expanded, torch.tensor(eps, device=Mx.device), torch.abs(diff))
    dk_dMx = -v * torch.sign(diff) * (safe_abs ** (v-1)) * k.unsqueeze(-1)
    dk_dMx = torch.where(zero_mask_expanded, torch.zeros_like(dk_dMx), dk_dMx)
    

    # Backprop through linear layer: ∂k/∂x = ∂k/∂(Mx) @ M
    dk_dx = dk_dMx@sqrtM  # (batch, m, d_in)
    print(f'{dk_dx.transpose(1,-1).shape=}, {alphas.shape=}')
    dk_dx_sum = dk_dx.transpose(1,-1)@alphas # nmd -> ndm, mc -> ndc
    return dk_dx_sum.transpose(1,-1) # ndc -> ncd    

def get_laplacian_gen_squared_grads(
    x: torch.Tensor, 
    z: torch.Tensor, 
    sqrtM: Union[torch.Tensor, None], 
    v: float, 
    L: float,
    alphas: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes dk/dx for the kernel k(Mx, z) = ∏ exp(-|(Mx)_i - z_i|^v)
    
    Args:
        x: Input tensor (n, d_in) or (d_in,)
        z: Input tensor (m, d_in) or (d_in,)
        sqrtM: Transformation vector (d_in,)
        v: Exponent parameter
        L: bandwidth
        eps: Numerical stability term
        
    Returns:
        Gradient tensor of shape:
        - (n, m, d_in) if x is 2D and z is 2D
        - (m, d_in) if x is 1D and z is 2D
        - (d_in,) if both are 1D
    """
    # Ensure 2D tensors
    x = x.unsqueeze(0) if x.dim() == 1 else x
    z = z.unsqueeze(0) if z.dim() == 1 else z

    n, d = x.shape
    m, d2 = z.shape
    assert d == d2, "Feature dimensions must match"
    
    if sqrtM is None:
        sqrtM = torch.ones(x.shape[1], device=x.device, dtype=x.dtype)
    sqrtM = sqrtM.view(1, -1)

    # Transform x through linear layer
    Mx = x * sqrtM / L
    z = z * sqrtM / L
    
    pdists = torch.cdist(Mx, z, p=v) ## ||X/L-Z/L||_p = (\sum_{i=1}^d (|xi-zi|/L)^p)^(1/p)
    pdists_p = pdists**v ## \sum_{i=1}^d (|xi-zi|/L)^p
    k = torch.exp(-1*pdists_p) ## \prod_{i=1}^d exp(-(|xi-zi|/L)^p)
    
    # Compute gradient components for ∂k/∂(Mx)
    zero_mask = (pdists < eps) # (n, m)
    
    diff = Mx.unsqueeze(1) - z.unsqueeze(0)
    zero_mask_expanded = zero_mask.unsqueeze(-1)
    safe_abs = torch.where(zero_mask_expanded, torch.tensor(eps, device=Mx.device), torch.abs(diff))
    dk_dMx = -v * torch.sign(diff) * (safe_abs ** (v-1)) * k.unsqueeze(-1)
    dk_dMx = torch.where(zero_mask_expanded, torch.zeros_like(dk_dMx), dk_dMx)

    # Backprop through linear layer: ∂k/∂x = ∂k/∂(Mx) @ M
    dk_dx = dk_dMx*sqrtM  # (batch, m, d_in)
    dk_dx_sum = dk_dx.transpose(1,-1)@alphas # nmd -> ndm, mc -> ndc
    dk_dx_sum = dk_dx_sum.transpose(1,-1) # ndc -> ncd 
    dk_dx_sum = dk_dx_sum.reshape(-1, d) # ncd -> (nc)d
    return (dk_dx_sum**2).sum(dim=0)

def get_laplace_gen_agop(
    x: torch.Tensor, 
    z: torch.Tensor, 
    sqrtM: Union[torch.Tensor, None], 
    L: float,
    v: float, 
    alphas: torch.Tensor,
    diag: bool = False
) -> torch.Tensor:
    

    if diag:
        squared_grads = get_laplacian_gen_squared_grads(x, z, sqrtM, v, L, alphas)
        return squared_grads
    else:
        grads = get_laplacian_gen_grad(x, z, sqrtM, v, L, alphas)
        grads = grads.reshape(-1, grads.shape[-1])
        agop = grads.T@grads
        return agop

'''
def euclidean_distances(samples, centers, center_batch_size=128, squared=True):
    """
    Computes the Summed Euclidean Squared Distance (N, M) by batching the 
    M (centers) dimension to conserve memory.

    Args:
        samples: (N, P, Q, K) tensor.
        centers: (M, P, Q, K) tensor.
        center_batch_size (int): Number of centers (M) to process simultaneously.
        squared (bool): If False, returns the true total distance.
    """
    
    M_total = centers.size(0)
    all_dists = []

    # Iterate over center batches
    for start_idx in range(0, M_total, center_batch_size):
        end_idx = min(start_idx + center_batch_size, M_total)
        centers_batch = centers[start_idx:end_idx]
        
        # 1. Expand samples (N, 1, P, Q, K) and centers_batch (1, M_batch, P, Q, K)
        # M_batch is small (e.g., 128), which keeps the intermediate 'diff' tensor small.
        samples_e = samples.unsqueeze(1)
        centers_e = centers_batch.unsqueeze(0)
        
        # diff: (N, M_batch, P, Q, K)
        diff = samples_e - centers_e 
        
        # 2. Compute (Si[p, q, k] - Cj[p, q, k])^2 and sums over k
        # dist_pq: || Si[p, q, :] - Cj[p, q, :] ||_2^2. Shape: (N, M_batch, P, Q)
        dist_pq = diff.pow(2).sum(dim=-1) 

        # Apply square root if requested
        if not squared:
            final_dist_batch = dist_pq.clamp(min=0).sqrt()
            
        # 3. Sum over P and Q dimensions
        # final_dist_batch: sumP( sumQ( dist_pq ) ). Shape: (N, M_batch)
        final_dist_batch = dist_pq.sum(dim=(-1, -2))     
            
        all_dists.append(final_dist_batch)
        
        # Explicitly clear temporary tensors to help garbage collection
        del centers_batch, samples_e, centers_e, diff, dist_pq, final_dist_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate all (N, M_batch) tensors back into a single (N, M_total) tensor
    final_dist = torch.cat(all_dists, dim=1)
    
    return final_dist

def euclidean_distances_M(samples, centers, M, center_batch_size=128, squared=True):
    """
    Computes the Summed Mahalanobis Squared Distance (N, M) by batching the 
    M (centers) dimension to conserve memory. This avoids creating the massive 
    (N, M, P, Q, K) difference tensor all at once.

    Metric: D_M^2(Si, Cj) = sumP( sumQ( (Si[p,q,:] - Cj[p,q,:])^T M (Si[p,q,:] - Cj[p,q,:]) ) )

    Args:
        samples: (N, P, Q, K) tensor.
        centers: (M, P, Q, K) tensor.
        M: Weighting matrix. Must be (K, K) or (K).
        center_batch_size (int): Number of centers (M) to process simultaneously.
        squared (bool): If False, returns the true total distance.
    """
    
    M_total = centers.size(0)
    all_dists = []
    
    # 1. Handle diagonal M case (1D tensor of weights) for compatibility
    # If M is 1D (M_diag), convert it to a full matrix for the matrix multiplication
    if M.dim() == 1:
        M = torch.diag_embed(M)

    # 2. Iterate over center batches
    for start_idx in range(0, M_total, center_batch_size):
        end_idx = min(start_idx + center_batch_size, M_total)
        centers_batch = centers[start_idx:end_idx]
        
        # Expand samples (N, 1, P, Q, K) and centers_batch (1, M_batch, P, Q, K)
        samples_e = samples.unsqueeze(1)
        centers_e = centers_batch.unsqueeze(0)
        
        # diff: (N, M_batch, P, Q, K) - M_batch is small to conserve memory
        diff = samples_e - centers_e 
        
        # 3. Calculate the weighted difference: d @ M 
        weighted_diff = diff @ M # (N, M_batch, P, Q, K)

        # 4. Calculate d^T M d (sum over K)
        # dist_pq: (N, M_batch, P, Q)
        dist_pq = (weighted_diff * diff).sum(dim=-1) 

        # 5. Apply square root if requested
        if not squared:
            dist_pq = dist_pq.clamp(min=0).sqrt()
            
        # 6. Sum over P and Q dimensions
        final_dist_batch = dist_pq.sum(dim=(-1, -2)) # (N, M_batch)
            
        all_dists.append(final_dist_batch)
        
        # Optional: Explicitly clear temporary tensors to help garbage collection
        del centers_batch, samples_e, centers_e, diff, weighted_diff, dist_pq, final_dist_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 7. Concatenate all batches back into a single (N, M_total) tensor
    final_dist = torch.cat(all_dists, dim=1)
    
    return final_dist
'''