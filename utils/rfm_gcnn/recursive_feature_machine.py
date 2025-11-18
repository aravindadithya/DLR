from utils.rfm_gcnn.eigenpro import KernelModel
    
import torch, numpy as np
from torchmetrics.functional.classification import accuracy
from utils.rfm_gcnn.generic_kernels import Kernel
from utils.rfm_gcnn.kernels import laplacian_M, gaussian_M, euclidean_distances_M, laplacian_gen, get_laplace_gen_agop, ntk_kernel
from tqdm.contrib import tenumerate
from utils.rfm_gcnn.utils import matrix_power_eigendecomposition, get_data_from_loader
import time

class RecursiveFeatureMachine(torch.nn.Module):
    """
    Main object for RFMs with sklearn style interface. Subclasses must implement the kernel and update_M methods. 
    The subclasses may be either specific kernels (Laplace, Gaussian, GeneralizedLaplace, etc.), in which case the kernel method is automatically derived,
    or generic kernels (GenericKernel), in which case a Kernel object must be provided. I.e. one can either define:
    ```python
        from rfm import LaplaceRFM
        model = LaplaceRFM(bandwidth=1, device='cpu', reg=1e-3, iters=3, bandwidth_mode='constant')

        or

        from rfm import GenericRFM
        from rfm.kernels_new import LaplaceKernel
        model = GenericRFM(kernel=LaplaceKernel(bandwidth=1, exponent=1.2), device='cpu', reg=1e-3, iters=3, bandwidth_mode='constant')
    ```
    """

    def __init__(self, layer, num_classes=10, device=torch.device('cpu'), mem_gb=8, diag=False, centering=False, reg=1e-3, iters=5, p_batch_size=None, bandwidth_mode='constant'):
        """
        :param device: device to run the model on
        :param mem_gb: memory in GB for AGOP
        :param diag: if True, Mahalanobis matrix M will be diagonal
        :param centering: if True, update_M will center the gradients before taking an outer product
        :param reg: regularization for the kernel matrix
        :param iters: number of iterations to run
        :param p_batch_size: batch size over centers for AGOP computation
        :param bandwidth_mode: 'constant' or 'adaptive'
        """
        super().__init__()
        self.M = None
        self.sqrtM = None
        self.use_sqrtM = False
        self.model = None
        self.diag = diag # if True, Mahalanobis matrix M will be diagonal
        self.centering = centering # if True, update_M will center the gradients before taking an outer product
        self.device = device
        self.mem_gb = mem_gb
        self.reg = reg # only used when fit with direct solve
        self.iters = iters
        self.kernel_type = None
        self.p_batch_size = p_batch_size
        self.agop_power = 0.5 # power for root of agop
        self.bandwidth_mode = bandwidth_mode
        self.max_lstsq_size = 70_000 # max number of points to use for direct solve
        self.layer = layer
        self.num_classes = num_classes

    def kernel(self, x, z):
        raise NotImplementedError("Must implement this method in a subclass")
    
    def update_M(self):
        raise NotImplementedError("Must implement this method in a subclass")
    
    def reset_adaptive_bandwidth(self):
        if self.kernel_type != 'generic':
            raise ValueError("Cannot reset bandwidth for non-generic kernels, choose constant bandwidth mode")
        self.kernel_obj._reset_adaptive_bandwidth()
        return 

    def tensor_copy(self, tensor):
        """
        Create a CPU copy of a tensor.
        :param tensor: Tensor to copy.
        :param keep_device: If True, the device of the original tensor is kept.
        :return: CPU copy of the tensor.
        """
        if self.keep_device or tensor.device.type == 'cpu':
            return tensor.clone()
        else:
            return tensor.cpu()

    def update_best_params(self, best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth, current_metric, current_iter):
        # if classification and accuracy higher, or if regression and mse lower
        if self.classification and current_metric > best_metric:
            best_metric = current_metric
            best_alphas = self.tensor_copy(self.weights)
            best_iter = current_iter
            best_bandwidth = self.bandwidth if self.kernel_type != 'generic' else self.kernel_obj.bandwidth+0
            if self.M is not None:
                best_M = self.tensor_copy(self.M)
                if self.use_sqrtM:
                    best_sqrtM = matrix_power(self.M, self.agop_power)
            else:
                best_M = None
                best_sqrtM = None
        elif not self.classification and current_metric < best_metric:
            best_metric = current_metric
            best_alphas = self.tensor_copy(self.weights)
            best_iter = current_iter
            best_bandwidth = self.bandwidth if self.kernel_type != 'generic' else self.kernel_obj.bandwidth+0
            if self.M is not None:
                best_M = self.tensor_copy(self.M)
                if self.use_sqrtM:
                    best_sqrtM = matrix_power(self.M, self.agop_power)
            else:
                best_M = None
                best_sqrtM = None

        return best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth
        
    def fit_predictor(self, centers, targets, classification=False, 
                      class_weight=None, bs=None, lr_scale=1, 
                      verbose=True, solver='solve', **kwargs):
        
        if self.bandwidth_mode == 'adaptive':
            # adaptive bandwidth will be reset on next kernel computation
            print("Resetting adaptive bandwidth")
            self.reset_adaptive_bandwidth()

        self.centers = centers
        if self.M is None:
            if self.diag:
                self.M = torch.ones(centers.shape[-1], device=self.device, dtype=centers.dtype)
            else:
                self.M = torch.eye(centers.shape[-1], device=self.device, dtype=centers.dtype)
        if self.fit_using_eigenpro:
            if self.prefit_eigenpro:
                random_indices = torch.randperm(centers.shape[0])[:self.max_lstsq_size]
                start = time.time()
                sub_weights = self.fit_predictor_lstsq(centers[random_indices], targets[random_indices], class_weight=class_weight, solver=solver)
                end = time.time()
                print(f"Time taken to prefit Eigenpro with {self.max_lstsq_size} points: {end-start} seconds")
                initial_weights = torch.zeros_like(targets)
                initial_weights[random_indices] = sub_weights.to(targets.device, dtype=targets.dtype)
            else:
                initial_weights = None

            self.weights = self.fit_predictor_eigenpro(centers, targets, bs=bs, lr_scale=lr_scale, 
                                                       verbose=verbose, classification=classification, 
                                                       initial_weights=initial_weights,
                                                       **kwargs)
        else:
            self.weights = self.fit_predictor_lstsq(centers, targets, class_weight=class_weight, solver=solver)

    def fit_predictor_lstsq(self, centers, targets, class_weight=None, solver='solve'):
        assert(len(centers)==len(targets))
        '''
        if centers.device != self.device:
            centers = centers.to(self.device)
            targets = targets.to(self.device)
            '''
        targets = targets.to(self.device)
        centers = centers.to(self.device)
        kernel_matrix = self.kernel(centers, centers) 
        kernel_matrix = kernel_matrix.to(self.device)
       

        if class_weight == 'inverse':
            flat_targets = targets.flatten().long()
            class_weights = {
                0: len(flat_targets) / (2 * (flat_targets == 0).sum()),
                1: len(flat_targets) / (2 * (flat_targets == 1).sum())
            }

            sample_weights = torch.ones_like(targets, dtype=kernel_matrix.dtype).flatten()
            sample_weights[flat_targets==0] = class_weights[0]
            sample_weights[flat_targets==1] = class_weights[1]
            sample_weights = sample_weights.to(device=self.device, dtype=kernel_matrix.dtype)
            
            W = torch.diag(sample_weights)
            kernel_matrix = W@kernel_matrix
            targets = W@targets

        if self.reg > 0:
            kernel_matrix.diagonal().add_(self.reg)
        
        if solver == 'solve':
            out = torch.linalg.solve(kernel_matrix, targets)
        elif solver == 'cholesky':
            L = torch.linalg.cholesky(kernel_matrix, out=kernel_matrix)
            out = torch.cholesky_solve(targets, L)
        elif solver == 'lu':
            P, L, U = torch.linalg.lu(kernel_matrix)
            out = torch.linalg.lu_solve(P, L, U, targets)
        else:
            raise ValueError(f"Invalid solver: {solver}")
        
        return out

    def fit_predictor_eigenpro(self, centers, targets, bs, lr_scale, verbose, initial_weights=None, **kwargs):
        n_classes = 1 if targets.dim()==1 else targets.shape[-1]
        ep_model = KernelModel(self.kernel, centers, n_classes, device=self.device)
        if initial_weights is not None:
            ep_model.weight = initial_weights.to(ep_model.weight.device, dtype=ep_model.weight.dtype)
        _ = ep_model.fit(centers, targets, verbose=verbose, mem_gb=self.mem_gb, bs=bs, lr_scale=lr_scale, **kwargs)
        return ep_model.weight


    def predict(self, samples):
        out = self.kernel(samples.to(self.device), self.centers.to(self.device)) @ self.weights.to(self.device)
        return out.to(samples.device)


    def fit(self, train_data, test_data, iters=None, method='lstsq', 
            classification=True, verbose=True, M_batch_size=None, 
            class_weight=None, return_best_params=False, bs=None, 
            return_Ms=False, lr_scale=1, total_points_to_sample=10000, 
            solver='solve', fit_last_M=False, prefit_eigenpro=True, 
            **kwargs):
        """
        :param train_data: torch.utils.data.DataLoader or tuple of (X, y)
        :param test_data: torch.utils.data.DataLoader or tuple of (X, y)
        :param iters: number of iterations to run
        :param method: 'lstsq' or 'eigenpro'
        :param classification: if True, the model will tune for (and report) accuracy, else just MSE loss
        :param verbose: if True, print progress
        :param M_batch_size: batch size over samples for AGOP computation
        :param class_weight: 'inverse' or None
        :param return_best_params: if True, return the best parameters
        :param bs: batch size for prediction
        :param return_Ms: if True, return the Mahalanobis matrix at each iteration
        :param lr_scale: learning rate scale for EigenPro
        :param total_points_to_sample: number of points to sample for AGOP computation
        :param solver: 'solve' or 'cholesky' or 'lu', used in LSTSQ computation
        :param fit_last_M: if True, fit the Mahalanobis matrix one last time after training
        :param prefit_eigenpro: if True, prefit EigenPro with a subset of <= max_lstsq_size samples
        """
                
        self.fit_using_eigenpro = (method.lower()=='eigenpro')
        self.prefit_eigenpro = prefit_eigenpro
        self.use_sqrtM = self.kernel_type in ['laplacian_gen', 'generic']
        self.classification = classification

        if iters is None:
            iters = self.iters

        if class_weight is not None and self.fit_using_eigenpro:
            raise ValueError("Class weights are not supported for EigenPro")

        if verbose and class_weight == 'inverse':
            print("Weighting samples by inverse class frequency")
        
        if isinstance(train_data, torch.utils.data.DataLoader):
            print("Loaders provided")
            X_train, y_train = get_data_from_loader(train_data, self.layer, self.num_classes)
            X_test, y_test = get_data_from_loader(test_data, self.layer, self.num_classes)
        else:
            X_train, y_train = train_data
            X_test, y_test = test_data

        self.keep_device = X_train.shape[1] > X_train.shape[0] # keep previous Ms on GPU if more features than samples

        mses, Ms = [], []
        best_alphas, best_M, best_sqrtM = None, None, None
        best_metric = float('inf') if not classification else 0 
        best_iter = None
        best_bandwidth = self.bandwidth if self.kernel_type != 'generic' else self.kernel_obj.bandwidth+0
        for i in range(iters):
            self.fit_predictor(X_train, y_train, X_val=X_test, y_val=y_test, 
                               classification=classification, class_weight=class_weight, 
                               bs=bs, lr_scale=lr_scale, verbose=verbose, solver=solver, **kwargs)
            
            if classification:
                test_acc = self.score(X_test, y_test, bs, metric='accuracy')
                if method == 'lstsq':
                    train_acc = self.score(X_train, y_train, bs, metric='accuracy')
                    if verbose:
                        print(f"Round {i}, Train Acc: {100*train_acc:.2f}%, Test Acc: {100*test_acc:.2f}%")
                else:
                    if verbose:
                        print(f"Round {i}, Test Acc: {100*test_acc:.2f}%")

            test_mse = self.score(X_test, y_test, bs, metric='mse')

            if verbose:
                print(f"Round {i}, Test MSE: {test_mse:.4f}")

            # if classification and accuracy higher, or if regression and mse lower
            if return_best_params:
                best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth = self.update_best_params(best_metric, best_alphas, 
                                                                                                                best_M, best_sqrtM, 
                                                                                                                best_iter, best_bandwidth, 
                                                                                                                test_acc if classification else test_mse, i)

            self.fit_M(X_train, y_train, verbose=verbose, M_batch_size=M_batch_size, 
                       use_sqrtM=self.use_sqrtM, total_points_to_sample=total_points_to_sample, 
                       **kwargs)
                        
            if return_Ms:
                Ms.append(self.M.cpu()+0)
                mses.append(test_mse)

        self.fit_predictor(X_train, y_train, X_val=X_test, y_val=y_test, 
                           class_weight=class_weight, verbose=verbose, 
                           classification=classification, bs=bs, **kwargs)
        final_mse = self.score(X_test, y_test, bs=bs, metric='mse')
        
        if verbose:
            print(f"Final MSE: {final_mse:.4f}")
        if classification:
            final_test_acc = self.score(X_test, y_test, bs=bs, metric='accuracy')
            if verbose:
                print(f"Final Test Acc: {100*final_test_acc:.2f}%")

        if return_best_params:
            best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth = self.update_best_params(best_metric, best_alphas, best_M, 
                                                                                                                best_sqrtM, best_iter, best_bandwidth, 
                                                                                                                final_test_acc if classification else final_mse, 
                                                                                                                iters)
            print(f"Returning best parameters with value: {best_metric:.4f}")
            self.M = None if best_M is None else best_M.to(self.device)
            if self.use_sqrtM and best_sqrtM is not None:
                self.sqrtM = best_sqrtM.to(self.device)
            else:
                self.sqrtM = None
            self.weights = best_alphas.to(self.device)
            if self.kernel_type == 'generic':
                self.kernel_obj.bandwidth = best_bandwidth
            else:
                self.bandwidth = best_bandwidth

        self.best_iter = best_iter
        if fit_last_M:
            self.fit_M(X_train, y_train, verbose=verbose, M_batch_size=M_batch_size, use_sqrtM=self.use_sqrtM, 
                        total_points_to_sample=total_points_to_sample, fit_last_M=fit_last_M, **kwargs)
            Ms.append(self.tensor_copy(self.M))

        if return_Ms and fit_last_M:
            self.agop_best_model = Ms[best_iter]

        if return_Ms:
            return Ms, mses
            
        return final_mse
    
    def _compute_optimal_M_batch(self, p_centers, c_classes, p_feat, q_feat, k_dim, scalar_size=4):
        """
        Computes the optimal batch size for AGOP when inputs are 4D (N, P_feat, Q_feat, K_dim).
        
        Args:
            p_centers (int): Number of centers (M_centers).
            c_classes (int): Number of classes (C).
            p_feat (int): Spatial dimension P.
            q_feat (int): Spatial dimension Q.
            k_dim (int): Feature dimension K (size of M).
            scalar_size (int): Size in bytes of the float type (e.g., 4 for float32).
        """
        THREADS_PER_BLOCK = 512 # pytorch default CUDA threads per block
        
        def tensor_mem_usage(numels):
            """Calculates memory footprint in units of THREADS_PER_BLOCK elements (approximate)."""
            return np.ceil(numels / THREADS_PER_BLOCK) * THREADS_PER_BLOCK

        def max_tensor_size(mem):
            """Calculates maximum possible tensor size (elements) given memory budget (bytes)."""
            # Calculates the max number of elements that fit a block-aligned memory chunk.
            return int(np.floor(mem / THREADS_PER_BLOCK) * (THREADS_PER_BLOCK / scalar_size))

        # Total feature elements per sample/center (P x Q x K)
        D_total = p_feat * q_feat * k_dim

        # 1. Calculate fixed memory usage (independent of batch size N)
        # M_mem uses K_dim
        M_mem_elements = k_dim if self.diag else k_dim**2
        
        # Centers memory usage: (M_centers, P_feat, Q_feat, K_dim)
        centers_mem_elements = p_centers * D_total
        
        M_mem = tensor_mem_usage(M_mem_elements)
        centers_mem = tensor_mem_usage(centers_mem_elements)
        
        # Safely determine current memory usage (assuming GPU usage)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            curr_mem_use = torch.cuda.memory_allocated() # in bytes
        else:
            curr_mem_use = 0
            
        # Total available memory for scaling tensors (G and intermediates)
        mem_available = (self.mem_gb * 1024**3) - curr_mem_use - (M_mem + centers_mem) * scalar_size
        
        # 2. Adapt the original AGOP memory formula structure (using D_total where size matters)
        
        # Non-scaling memory overhead (based on the original AGOP formula)
        # Assuming the original 'd' was the total feature size D_total for this term
        non_scaling_mem = (
            3 * tensor_mem_usage(p_centers) * scalar_size + 
            tensor_mem_usage(p_centers * c_classes * D_total) * scalar_size
        )
        
        # Denominator represents memory per sample (N=1) that scales with batch size
        # Original: (2*scalar_size*(1+p))
        scaling_factor = (2 * scalar_size * (1 + p_centers)) 
        
        # Calculate optimal batch size elements
        M_batch_size_elements = (mem_available - non_scaling_mem) / scaling_factor

        # Convert elements back to batch size using max_tensor_size logic
        M_batch_size = max_tensor_size(M_batch_size_elements * scalar_size)
        
        # Safety check: batch size must be at least 1
        return max(1, int(M_batch_size))
        
    def fit_M(self, samples, labels, p_batch_size=None, M_batch_size=None,
              verbose=True, total_points_to_sample=50000, use_sqrtM=False, **kwargs):
        """
        Applies AGOP to update the Mahalanobis matrix M for 4D inputs (N, P, Q, K).
        
        The 'update_M' method is assumed to handle the 4D input correctly.
        """
        
        # 1. Extract 4D and center/class dimensions
        n, p_feat, q_feat, k_dim = samples.shape
        p_centers = self.centers.shape[0] # Number of centers/prototypes
        c_classes = labels.shape[-1]      # Number of classes (assuming labels is one-hot)

        # 2. Initialize M
        # The M matrix is K x K or K
        M = torch.zeros_like(self.M) if self.M is not None else (
            torch.zeros(k_dim, dtype=samples.dtype) if self.diag else torch.zeros(k_dim, k_dim, dtype=samples.dtype))
        
        # 3. Determine Batch Size
        if M_batch_size is None:
            # Use samples.element_size() as a fallback if self.M is None
            BYTES_PER_SCALAR = self.M.element_size() if self.M is not None else samples.element_size()
            
            # Recalculate optimal batch size using 4D dimensions
            M_batch_size = self._compute_optimal_M_batch(
                p_centers, c_classes, p_feat, q_feat, k_dim, scalar_size=BYTES_PER_SCALAR
            )

            if verbose:
                print(f"Computed optimal M batch size: {M_batch_size}")
                
        # Safety check for batch size
        if M_batch_size <= 0:
            M_batch_size = 1 # Fallback to smallest valid batch size

        # 4. Prepare Batches for Iteration
        batches = torch.randperm(n).split(M_batch_size)

        num_batches = 1 + total_points_to_sample // M_batch_size
        batches = batches[:num_batches]
        if verbose:
            print(f'Sampling AGOP on maximum of {num_batches * M_batch_size} total points')

        # 5. Iterate and Accumulate M
        # Only import tenumerate if verbose is True
        if verbose:
            try:
                from tqdm.auto import tqdm as tenumerate
            except ImportError:
                def tenumerate(iterable, desc=None, **kwargs): return enumerate(iterable)
            
            for i, bids in tenumerate(enumerate(batches), desc="Updating M"):
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                # p_batch_size is kept for compatibility with the original signature
                M.add_(self.update_M(samples[bids], p_batch_size))

        else:
            # The non-verbose loop
            for bids in batches:
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                M.add_(self.update_M(samples[bids], p_batch_size))
            
        # 6. Finalize M
        M_max = M.max()
        self.M = M / M_max if M_max > 0 else M
        
        if use_sqrtM:
            # Assuming matrix_power and agop_power are defined in the class context
            if hasattr(self, 'matrix_power') and hasattr(self, 'agop_power') and callable(getattr(self, 'matrix_power', None)):
                self.sqrtM = matrix_power(self.M, self.agop_power)
            else:
                if verbose:
                    print("Warning: use_sqrtM is True but matrix_power/agop_power are undefined or non-callable.")
        
        del M
        
        if verbose:
            print("M update complete.")
   

        
    def score(self, samples, targets, bs, metric='mse'):
        """
        samples: torch.Tensor of shape (n, d)
        targets: torch.Tensor of shape (n, c)
        bs: batch size over samples for prediction
        metric: 'mse' or 'accuracy'
        """
        if bs is None:
            preds = self.predict(samples.to(self.device)).to(targets.device)
        else:
            all_preds = []
            for batch in samples.split(bs):
                preds = self.predict(batch.to(self.device)).to(targets.device)
                all_preds.append(preds)
            preds = torch.cat(all_preds, dim=0).to(targets.device)
        if metric=='accuracy':
            if preds.shape[-1]==1:
                num_classes = len(torch.unique(targets))
                if num_classes==2:
                    preds = torch.where(preds > 0.5, 1, 0).reshape(targets.shape)
                    return accuracy(preds, targets, task="binary").item()
                else:
                    return accuracy(preds, targets, task="multiclass", num_classes=num_classes).item()
            else:
                preds_ = torch.argmax(preds,dim=-1)
                targets_ = torch.argmax(targets,dim=-1)
                return accuracy(preds_, targets_, task="multiclass", num_classes=preds.shape[-1]).item()
        
        elif metric=='mse':
            return (targets - preds).pow(2).mean()

class GenericRFM(RecursiveFeatureMachine):
    """
    The preferred RFM subclass for generic kernels. These enable more fine-grained control over the kernel functions
    including adaptive bandwidths and different exponents. The kernel functions are faster and more memory efficient than
    the specific RFM subclasses.
    """
    def __init__(self, kernel: Kernel, agop_power=0.5, **kwargs):
        super().__init__(**kwargs)
        self.kernel_obj = kernel
        self.kernel_type = 'generic'
        self.agop_power = agop_power
        
    def kernel(self, x, z):
        return self.kernel_obj.get_kernel_matrix(x, z, self.sqrtM)

    def update_M(self, samples, p_batch_size):
        if self.M is None:
            if self.diag:
                self.M = torch.ones(samples.shape[-1], device=samples.device, dtype=samples.dtype)
                self.sqrtM = torch.ones(samples.shape[-1], device=samples.device, dtype=samples.dtype)
            else:
                self.M = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)
                self.sqrtM = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)

        samples = samples.to(self.device)
        self.centers = self.centers.to(self.device)
        agop_func = self.kernel_obj.get_agop_diag if self.diag else self.kernel_obj.get_agop
        agop = agop_func(x=self.centers, z=samples, coefs=self.weights.t(), mat=self.sqrtM)
        return agop

class LaplaceRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1., **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel_type = 'laplace'

    def kernel(self, x, z):
        return laplacian_M(x, z, self.M, self.bandwidth)
    
    def update_M(self, samples, p_batch_size):
        samples = samples.to(self.device)
        self.centers = self.centers.to(self.device)
        """Performs a batched update of M."""
        K = self.kernel(samples, self.centers)
        if p_batch_size is None: 
            p_batch_size = self.centers.shape[0]
        
        dist = euclidean_distances_M(samples, self.centers, self.M, squared=False)
        dist = torch.where(dist < 1e-10, torch.zeros(1, device=dist.device).float(), dist)

        K.div_(dist)
        del dist
        K[K == float("Inf")] = 0.0

        p, d = self.centers.shape
        p, c = self.weights.shape
        n, d = samples.shape

        samples_term = (K @ self.weights).reshape(n, c, 1)  # (n, p)  # (p, c)

        if self.diag:
            temp = 0
            for p_batch in torch.arange(p).split(p_batch_size):
                # temp[j,l,d] += \sum_i M[j,i] * coefs[i,l] * x[i,d]
                temp += K[:, p_batch] @ ( # (n, len(p_batch))
                    self.weights[p_batch,:].view(len(p_batch), c, 1) * (self.centers[p_batch,:] * self.M).view(len(p_batch), 1, d)
                ).reshape(
                    len(p_batch), c * d
                )  # (len(p_batch), cd)
            
            centers_term = temp.view(n, c, d)

            # M[j, i] * coefs[i, l] * z[j, d]
            samples_term = samples_term * (samples * self.M).reshape(n, 1, d)

        else:
            temp = 0
            for p_batch in torch.arange(p).split(p_batch_size):
                temp += K[:, p_batch] @ ( # (n, len(p_batch))
                    self.weights[p_batch,:].view(len(p_batch), c, 1) * (self.centers[p_batch,:] @ self.M).view(len(p_batch), 1, d)
                ).reshape(
                    len(p_batch), c * d
                )  # (len(p_batch), cd)
            
            centers_term = temp.view(n, c, d)

            samples_term = samples_term * (samples @ self.M).reshape(n, 1, d)

        G = (centers_term - samples_term) / self.bandwidth  # (n, c, d)

        del centers_term, samples_term, K
        
        if self.centering:
            G = G - G.mean(0) # (n, c, d)
        
        # return quantity to be added to M. Division by len(samples) will be done in parent function.
        if self.diag:
            return torch.einsum('ncd, ncd -> d', G, G)
        else:
            return torch.einsum("ncd, ncD -> dD", G, G)
        

class GaussRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1., **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel_type = 'gaussian'

    def kernel(self, x, z):
        return gaussian_M(x, z, self.M, self.bandwidth)

    def update_M(self, samples, p_batch_size=None):
        #TODO: Compute the closed form. Optional tho
        
        # 1. Setup Tensors and Gradient Tracking
        samples = samples.to(self.device).requires_grad_(True)
        self.centers = self.centers.to(self.device)
        self.M = self.M.to(self.device)
        self.weights = self.weights.to(self.device)

        # Get dimensions
        n, p_feat, q_feat, k_dim = samples.shape
        m_centers = self.centers.shape[0]
        c_classes = self.num_classes 
        
        # 2. Compute the function output L = K @ W
        # K = (N, M_centers)
        K = self.kernel(samples, self.centers) 
        # L = (N, M_centers) @ (M_centers, C) -> (N, C)
        L = K @ self.weights                   

        # 3. Compute the Gradient Term G using autograd
        
        # G_auto must have shape (n, c, p_feat, q_feat, k_dim)
        G_auto = torch.zeros(n, c_classes, p_feat, q_feat, k_dim, device=self.device)
        
        for c_idx in range(c_classes):
            # Compute the gradient of the c-th column of L w.r.t. all samples.
            
            # The result grad_L_c will match the shape of 'samples': (N, P, Q, K)
            grad_L_c = torch.autograd.grad(
                outputs=L[:, c_idx],
                inputs=samples,
                grad_outputs=torch.ones_like(L[:, c_idx]),
                retain_graph=True, 
                allow_unused=True
            )[0]
            
            # The manual calculation of G is proportional to the NEGATIVE gradient of K@W
            # Use '...' to assign the (N, P, Q, K) tensor into G_auto's N, P, Q, K slices
            #G_auto[:, c_idx, ...] = -grad_L_c
            G_auto[:, c_idx, ...] = grad_L_c
        
        # 4. Apply Scaling Factor
        # This is a scalar division applied to the entire 5D tensor G_auto
        G = G_auto / self.bandwidth**2 

        '''
        # 5. Apply Centering (Same as original manual code)
        if self.centering:
            # Mean is computed over the N dimension, resulting in (1, C, P, Q, K)
            G = G - G.mean(0) 
        '''
        '''
        # 6. Final Accumulation for M update (Sum over N, C, P, Q)
        if self.diag:
            # Result: Sum_{n, c, p, q} (G_{ncpqk} * G_{ncpqk}) -> k (diagonal of M)
            return torch.einsum('ncpqk, ncpqk -> k', G, G)
        else:
            # Result: Sum_{n, c, p, q} (G_{ncpqk} * G_{ncpqK}) -> kK (full M matrix)
            return torch.einsum("ncpqk, ncpqK -> kK", G, G)
        '''
            
        return torch.einsum("ncpqk, ncpqK -> kK", G, G)

        


