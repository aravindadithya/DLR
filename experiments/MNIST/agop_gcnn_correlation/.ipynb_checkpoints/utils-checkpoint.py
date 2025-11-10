from torch.linalg import norm, svd
from matplotlib import pyplot as plt
import math
import torch.distributions as distributions
import cvxpy as cp
import numpy as np
plt.switch_backend('Agg')
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
    
def sqrt(G):
    U, s, Vt = svd(G)
    s = torch.pow(s, 1./2)
    G = U @ torch.diag(s) @ Vt
    return G



def vis(tensor_input, max_channels, fname, title ="Tensor Channel View"):
    """
    Visualizes a single image tensor (C x H x W) by plotting each channel.
    Handles tensors where C > 3.
    """
    
    if torch.is_tensor(tensor_input):
        # Detach, move to CPU, and convert to NumPy
        data = tensor_input.detach().cpu().numpy()
    else:
        data = tensor_input

    # Ensure we are in C x H x W format. If N x C x H x W (batch size 1), squeeze the batch dim.
    if data.ndim == 4 and data.shape[0] == 1:
        data = data.squeeze(0)
    
    if data.ndim != 3:
        print(f"Error: Expected 3 dimensions (C, H, W), got {data.ndim}. Aborting visualization.")
        return

    C, H, W = data.shape
    channels_to_show = min(C, max_channels)
    
    # Calculate grid dimensions (e.g., 9 channels -> 3x3 grid)
    cols = math.ceil(math.sqrt(channels_to_show))
    rows = math.ceil(channels_to_show / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(f"{title} ({C} Channels Total, Showing {channels_to_show})", fontsize=12)

    # Handle cases where axes is not a 2D array (e.g., if rows=1, cols=1)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    # Determine common scale for all channels (important for feature maps)
    min_val = data.min()
    max_val = data.max()

    for i in range(channels_to_show):
        ax = axes[i]
        # Display the i-th channel as a grayscale map
        # Using a fixed color scale (vmin/vmax) shows relative feature intensity
        ax.imshow(data[i], cmap='viridis', vmin=min_val, vmax=max_val)
        ax.set_title(f"Channel {i}", fontsize=10)
        ax.axis('off')

    # Remove unused subplots
    for j in range(channels_to_show, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'images/{fname}')
    plt.show()

def sample_normal(covariance_matrix_M, num_rows_k, matrix_dim_t):
   
    # Create a mean vector of zeros, compatible with the dimensions of M
    dtype = covariance_matrix_M.dtype
    mean_vector = torch.zeros(matrix_dim_t, dtype=dtype, device=device)
    mean_vector.to(device)
    #covariance_matrix_M.to(device)
    # Create a multivariate normal distribution object
    # torch.distributions.MultivariateNormal expects a covariance_matrix.
    # It internally checks for positive definiteness.
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

def find_covariance_matrix_m(S_target_np, P_matrices_np, M_dim, solver_name='SCS'):

    M = cp.Variable((M_dim, M_dim), symmetric=True)
    sum_term = 0
    for P_i in P_matrices_np:
        sum_term += P_i.T @ M @ P_i

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