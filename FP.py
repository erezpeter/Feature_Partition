from joblib import Parallel, delayed
from scipy.sparse import coo_matrix


import numpy as np
import scipy
import scipy.spatial
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.manifold import _utils
from sklearn.neighbors import NearestNeighbors


import new_binary_search_perplexity 

# Verifies that the input is a valid SVD decomposition (U, D, VT).
def verify_svd(x):
    if len(x) != 3:
        raise ValueError("Expected a tuple (U, D, VT) representing the SVD decomposition.\n"+\
            "U should be of shape (N, r), D of shape (r,), and VT of shape (r, D)." )
    U, D, VT = x
    r = len(D)
    
    if len(np.shape(U)) != 2:
        raise ValueError("U (left singular vectors) must be a 2D array of shape (N, r).")

    if len(np.shape(D)) != 1:
        raise ValueError("D (singular values) must be a 1D array of shape (r,).")

    if len(np.shape(VT)) != 2:
        raise ValueError("VT (right singular vectors) must be a 2D array of shape (r, D).")

    if np.shape(U)[1] != r or np.shape(VT)[0] != r:
        raise ValueError("Dimension mismatch: U.shape[1] and VT.shape[0] must match the length of D.")

        
def throw_exception(val, var_name, bigger_than_val=None):
    if bigger_than_val is not None:
        if val<= bigger_than_val:
            raise Exception(var_name+' should be bigger than '+str(bigger_than_val)+'.')

# Computes the negative entropy of a (row- or column-) stochastic matrix along the specified axis.
#
# Parameters:
#   A : numpy.ndarray
#       Non-negative array containing probabilities along the specified axis.
#   axis : int or None, optional
#       Axis along which to compute the negative entropy. If None, treats A as a flat vector.
#
# Returns:
#   C : numpy.ndarray or scalar
#       The negative entropy computed along the specified axis. Returns a scalar if A is 1D or axis is None.
def FP_calc_neg_entropy(A,axis=None):        
    B = np.finfo(float).eps+A    
    
    if axis is None:
        
        C= np.sum(np.log(B)*B)
        
        return C
    C= np.sum(np.log(B)*B,axis=axis)
    
    return C


# Computes pairwise distances over samples using a diagonal metric.
#
# Parameters:
#   x : numpy.ndarray or tuple
#       - If ndarray: shape (N, D), where N is the number of samples and D is the dimension.
#       - If tuple: SVD decomposition of the data, given as (U, S, Vh) with shapes:
#         U: (N, S), S: (S,), Vh: (S, D),
#         where S is the rank of the SVD.
#
#   diag_metric : numpy.ndarray, optional
#       1D array of length D specifying weights for each dimension.
#       Computes distances as: 
#           ||y_i - y_j||^2_{a} = sum_{d=1}^D a[d] * (y_i[d] - y_j[d])^2,
#       where a = diag_metric.
#
# Returns:
#   dist : numpy.ndarray
#       Pairwise distance matrix of shape (N, N).
def get_dist(x,diag_metric=None, is_exact=True, n_neighbors=None):

    is_x_svd= isinstance(x,tuple)

    if is_x_svd:
        N= len(x[0])
        D= x[2].shape[-1]
    else:
        N,D = np.shape(x)
    

    if diag_metric is None:
        diag_metric= np.ones((x.shape[1]))
            
    if len(diag_metric.shape)>1:
        diag_metric= diag_metric.flatten()

    
    if is_x_svd:
        U, D, VT = x

        eigs, vecs= np.linalg.eig((VT*diag_metric[None,:])@VT.T)
        eigs= np.real(eigs)
        vecs= np.real(vecs)
        eigs[eigs<0.]=0.
        eigs = np.sqrt(eigs)
        
        distances= pairwise_distances( (U*D[None,:]) @(vecs*eigs[None,:]), squared=True)
        
    else:
        metric= np.sqrt(diag_metric[None,:])
        metric[metric<0.]=0.
        distances= pairwise_distances(x*metric,squared=True)


    return distances
        
        
    
def softmax(array, div=None):
    # Shift array by max for numerical stability
    if div is None:
        t= np.exp(array-np.max(array))
    else:
        t= np.exp((array-np.max(array))/div)
    
    sum_t= np.sum(t)
    
    # If the sum is too small (underflow), use uniform distribution
    if sum_t < np.finfo(float).eps:
        t=t*0+1
        sum_t= np.sum(t)
    return t/sum_t


from sklearn.decomposition import PCA as PCA_sklearn
# Computes the leading Singular Value Decomposition components of the input data matrix after centering 
# (subtracting the mean across samples).
#
# Parameters:
#   x : numpy.ndarray
#       Input data matrix of shape (N, D), where N is the number of samples and D is the number of features.
#   r : int
#       Number of leading SVD components to compute and return.
#
# Returns:
#   U : numpy.ndarray
#       Left singular vectors of shape (N, r).
#   S : numpy.ndarray
#       Singular values as a 1D array of length R.
#   VT : numpy.ndarray
#       Right singular vectors of shape (R, D).
def SVD_mean_shifted(data,r):
    pca=  PCA_sklearn(n_components=r)
    pca.fit(data)
    D = pca.singular_values_+0.
    VT = pca.components_+0.
    transformed_data = pca.transform(data)
    U = transformed_data / D[None,:]
    return (U,D, VT)


########################################################

import numpy as np
import math


OMEGA_TOLERANCE = 1e-8
W_TOLERANCE = 1e-8

def verify_omega(omega):
    if len(omega.shape)!=2:
        raise Exception('omega should be a 2-d tensor of sizes KxD, where K is the amount of partitions and D is the amount of features.\n'\
                        +' Current omega tensor is of shape '+omega.shape)
    sum_dims= np.sum(omega,axis=0)
    if np.max(sum_dims)>1+OMEGA_TOLERANCE:
        raise Exception('The sum over omega should be 1. One of the partition coordinates sums to '+str(np.max(sum_dims)))
    
    if np.min(sum_dims)<1-OMEGA_TOLERANCE:
        raise Exception('The sum over omega should be 1. One of the partition coordinates sums to '+str(np.min(sum_dims)))
        

def verify_W(W):
    if not isinstance(W, list) and not isinstance(W, np.ndarray):
        raise Exception('W should be a list of 2-d tensor of size NxN, or 3-d tensor of size KxNxN. Current tensor is of type '+type(W))
        
        
    for i in range(len(W)):
        if len(W[i].shape)!=2:
            raise Exception('W should be a 3-d tensor of sizes KxNxN, where K is the amount of partitions and N is the amount of points.\n'\
                            +' Current W tensor in its '+str(i)+' position is of shape '+W[i].shape)
        
        if isinstance(W[i],np.ndarray):           
            sum_dims= np.sum(W[i],axis=-1)
        else:
            sum_dims= W[i].sum(axis=-1)                   
                   
        if np.max(sum_dims)>1+W_TOLERANCE:
            raise Exception('The sum over the rows of W should be 1. One of the partition coordinates sums to '+str(np.max(sum_dims)))
        if np.min(sum_dims)<1-W_TOLERANCE:
            raise Exception('The sum over the rows of W should be 1. One of the partition coordinates sums to '+str(np.min(sum_dims)))

# Given data and feature partitions, computes optimal affinity matrices for each partition.
#
# Parameters:
#   x : numpy.ndarray or tuple
#       - If ndarray: shape (N, D), input data matrix with N samples and D features.
#       - If tuple: SVD decomposition (U, S, VT) with shapes:
#         U: (N, S), S: (S,), VT: (S, D), where S is the SVD rank.
#
#   omega : numpy.ndarray
#       Array of shape (K, D), where K is the number of partitions and D is the number of features.
#       Each row defines the weights for one feature partition.
#
#   perplexity : float
#       Positive scalar specifying the target perplexity for the affinity matrices.
#
#   is_parallel : bool
#       Whether to parallelize computations across blocks of samples.
#
#   block_size : int
#       If `is_parallel` is True, defines the number of samples processed per block.
#       If block_size >= N, disables block parallelism.
#
# Returns:
#   W : numpy.ndarray
#       Array of shape (K, N, N), where each W[k] is the affinity matrix of shape (N, N)
#       corresponding to the k-th feature partition.
    
def FP_get_W(x, omega, perplexity, is_parallel=True, block_size=100):
    
    is_x_svd= isinstance(x,tuple)

    if is_x_svd:
        N= len(x[0])
    else:
        N= len(x)
    K= len(omega)
    
    
    W= np.zeros((K, N, N),dtype=np.double)
    new_inv_epsilons = np.zeros((K,N),dtype=np.double)
    
    for s in range(K):

        if np.all(omega[[s],:]<1e-10):
            W[s,:,:]= 1/N
            continue
            
        cur_dist = get_dist(x,diag_metric= omega[s,:] )


        if is_parallel: 
            
            new_func = lambda i:  new_binary_search_perplexity._new_binary_search_perplexity(cur_dist[i: i+block_size].copy(),\
                                                                    float(perplexity),  i)

            ret = Parallel(n_jobs=-1)(delayed(new_func)(j) for j in range( 0, N, block_size))
            new_inv_epsilons[s] = np.concatenate([term for _,term in ret])
            W[s] = np.concatenate([term for term,_ in ret],axis=0)

        else:
            W[s], inv_epsilons[s]= new_binary_search_perplexity._new_binary_search_perplexity(\
                                                                    cur_dist,float(perplexity),  -1)
            
    
    verify_W(W)

    
    return W, new_inv_epsilons

    

# Given data and fixed affinity matrices, computes the optimal feature partitions.
#
# Parameters:
#   x : numpy.ndarray or tuple
#       - If ndarray: shape (N, D), input data matrix with N samples and D features.
#       - If tuple: SVD decomposition (U, S, VT) with shapes:
#         U: (N, S), S: (S,), VT: (S, D), where S is the SVD rank.
#
#   W : numpy.ndarray
#       Array of shape (K, N), containing affinity weights for each of the K partitions across N samples.
#
#   delta : float
#       Non-negative scalar specifying the regularization coefficient.
#
#
# Returns:
#   omega : numpy.ndarray
#       Array of shape (K, D), where each row contains non-negative weights assigned to features
#       for the corresponding partition. Each row sums to one:
#           omega[k, :].sum() == 1  for all k.
def FP_get_omega(x, W, delta=0.):

    is_x_svd= isinstance(x,tuple)

    if is_x_svd:
        N= np.shape(x[0])[0]
        dim= np.shape(x[2])[1]
    else:
        N, dim= np.shape(x)
    K= len(W)
    
    
    omega= np.zeros((K,dim))

    #
    # Computes for each feature d and partition k:
    #     (1/N) * sum_{i,j} W^{(k)}_{i,j} * (y_i[d] - y_j[d])^2
    #
    # Uses the quadratic form identity:
    #     sum_{i,j} W^{(k)}_{i,j} (y_i[d] - y_j[d])^2
    #     = y_d^T L y_d
    # where:
    #     y_d = [y_1[d], ..., y_N[d]]
    #     L = diag( sum( W^{(k)}, axis=0) + sum( W^{(k)}, axis=0)) - W^{(k)} - (W^{(k)})^T
    #
    if is_x_svd:
        U, D, VT= x
        for s in range(K):
            
            projected_W = -(W[s]+W[s].T)
            projected_W[np.arange(N),np.arange(N)] += np.sum(W[s],axis=1)+np.sum(W[s],axis=0)
            projected_W = U.T @ projected_W @ U


            projected_X = ((D/np.sqrt(N))[:,None]* VT)

            omega[s] = np.sum( (projected_W @ projected_X ) * projected_X,axis=0)
            
            
    else:
        for s in range(K):
            cur_res= np.mean(  x**2 *( np.sum(W[s],axis=0)+  np.sum(W[s],axis=1))[:,None], axis= 0 )
            omega[s]= cur_res-  2*np.mean(  (W[s]@ x)*x,axis=0)
                



    # If regularization is used (delta > 0), omega is computed via a softmax over the values above,
    # yielding smooth (probabilistic) feature weights. Otherwise (delta = 0), a hard assignment is performed.
    if delta>0.:
        for d in range(dim):
            omega[:,d] = softmax(-omega[:,d],div=delta)
        omega[omega<0.]=0.
        
    else:
        # When delta = 0, the regularized formulation reduces to the original problem.
        # Each feature is assigned to exactly one partition, with ties broken randomly.
        # Cases where a feature could belong to multiple partitions are rare and yield similar scores,
        # so the choice among them does not substantially affect the solution.

        for d in range(dim):
            min_inds = np.where(omega[:,d]==np.min(omega[:,d]))[0]
            if len(min_inds)>1:
                min_inds= np.random.RandomState().choice(min_inds)
            else:
                min_inds= min_inds[0]
            
            omega[:,d]= 0.    
            omega[min_inds,d]= 1.
    
    verify_omega(omega)
    return omega



# Computes the objective value of either the regularized problem (if delta > 0)
# or the original optimization problem (if delta = 0).
#
# Parameters:
#   x : numpy.ndarray or tuple
#       - If ndarray: shape (N, D), input data matrix with N samples and D features.
#       - If tuple: SVD decomposition (U, S, VT) with shapes:
#         U: (N, S), S: (S,), VT: (S, D), where S is the SVD rank.
#
#   W : numpy.ndarray
#       Array of shape (K, N, N), with non-negative values. Each W[k] is the affinity matrix 
#       for the k-th partition over N samples.
#
#   omega : numpy.ndarray
#       Array of shape (K, D), with non-negative values, where each row contains the feature weights 
#       for a partition. Each row sums to one.
#
#   perplexity : float
#       Positive scalar specifying the perplexity parameter.
#
#   delta : float
#       Non-negative scalar specifying the regularization coefficient.
#
#   get_separated_terms : bool
#       If True, returns both the smoothness score (objective) and the regularization term separately.
#       If False, returns only their sum.
#
# Returns:
#   score : float
#       Non-negative scalar. If get_separated_terms=True, this is the smoothness score (objective 
#       without regularization). If False, this is the total objective (smoothness + regularization).
#
#   entropy_omega : float
#       Non-negative scalar representing the regularization term. Returned only if get_separated_terms=True.
#
def FP_get_val(x, W, omega, perplexity, delta, get_seperated_terms=False):

    is_x_svd= isinstance(x,tuple)
                
    K= len(W)
    if is_x_svd:
        N=np.shape(x[0])[0]
        dim=np.shape(x[2])[1]
    else:
        N,dim=np.shape(x)
        
    # Computes for each feature d and partition k:
    #     (1/N) * sum_{k} sum_{i,j} \sum_{d} W^{(k)}_{i,j} * (y_i[d] - y_j[d])^2 \omega^{(k)}_d
    #
    # Uses the quadratic form identity:
    #     sum_{i,j} W^{(k)}_{i,j} (y_i[d] - y_j[d])^2
    #     = y_d^T L y_d
    # where:
    #     y_d = [y_1[d], ..., y_N[d]]
    #     L = diag(W^{(k)}.sum(axis=0) + W^{(k)}.sum(axis=1)) - W^{(k)} - (W^{(k)})^T
    #
    smooth_term =[]
    if is_x_svd:
        U,D,VT= x
        for s in range(K):
            projected_W = -W[s] -W[s].T
            projected_W[np.arange(N),np.arange(N)] += np.sum(W[s],axis=1)+ np.sum(W[s],axis=0)
            projected_W = U.T@projected_W@U

            projected_x= ((D/np.sqrt(N))[:,None]*VT)

            smooth_term.append( np.sum( (projected_W@projected_x)* projected_x*omega[s]))

    else:
                           
        for s in range(K):
            cur_res= np.mean( np.sum( x**2*omega[s,None,:] *( np.sum(W[s],axis=0)+  np.sum(W[s],axis=1))[:,None], axis=1 ) )
            cur_res-=  2*np.mean( np.sum( ( W[s]@ x)*x*omega[s], axis=1 ) )
            smooth_term.append( cur_res )
    
    smooth_term= np.sum(smooth_term)

    
    entropy_omega=0.
    if delta>0.:
        entropy_omega = np.finfo(float).eps+ omega
        entropy_omega = delta*(np.sum(FP_calc_neg_entropy(entropy_omega,axis=0))+dim*np.log(K))

        
    if get_seperated_terms:
        return smooth_term, entropy_omega
    return smooth_term+ entropy_omega
    

# Computes the initial regularization coefficient that defines the first 
# optimization problem to be solved by the algorithm.
#
# Input:
#   x : array-like or tuple
#       1. If ndarray: shape (N, D), where N is the number of samples and D is the dimension.
#       2. If tuple: SVD decomposition of the data with structure (U, S, VT):
#            - U: (N, S), left singular vectors
#            - S: (S,), singular values
#            - VT: (S, D), right singular vectors
#
#   K : int
#       Number of partitions.
#
#   perplexity : float
#       Positive scalar specifying the perplexity parameter.
#
#   is_parallel : bool
#       Whether to parallelize computations during the construction of affinity matrices (W).
#
#   block_size : int
#       Positive integer ≤ N. Defines the number of samples per parallel block when computing W 
#       (via FP_get_W()). If block_size ≥ N, parallelization is effectively disabled.
#
# Output:
#   delta_init : float
#       Non-negative scalar indicating the initial regularization coefficient to be used in the optimization.
#
def get_delta_init(x, K, perplexity, is_parallel=None, block_size= None):
    is_x_svd= isinstance(x,tuple)
    if is_x_svd:
        dim= np.shape(x[2])[1]
    else:
        dim= np.shape(x)[1]
    
    init_omega = np.ones((K,dim))/K
    W,_ = FP_get_W(x, init_omega, perplexity, is_parallel= is_parallel, block_size= block_size)
    laplacian_score = FP_get_val(x, W, init_omega, perplexity, delta=0., get_seperated_terms=True)[0]
    return laplacian_score/ (dim*np.log(K))
    
#########################################################


class FP:

    # Constructor for the FP class.
    #
    # NOTE: The computed objective score in this implementation is the original score 
    #       shown in the paper, divided by N (number of samples).
    #
    # Input parameters:
    #
    ### Optimization problem options:
    #
    #   K : int
    #       Number of partitions.
    #
    #   perplexity : float
    #       Positive scalar specifying the perplexity parameter.
    #
    #### Regularization options:
    #
    #   delta_steps : int (optional)
    #       Positive integer indicating the number of regularized optimization steps 
    #       before reaching the non-regularized feature partition problem. 
    #       If delta_steps=0, the non-regularized problem is solved directly.
    #
    #   delta_degrad : float (optional)
    #       Strictly positive scalar specifying the factor by which delta decreases at each step.
    #
    #   delta_init : float (optional)
    #       Non-negative scalar to manually set the initial delta instead of using the paper's heuristic.
    #       NOTE: Use of this parameter is generally not recommended.
    #
    #   delta_list : list of floats (optional)
    #       List of decreasing non-negative deltas ending in zero, specifying the sequence of 
    #       regularization parameters. 
    #       NOTE: Use of this parameter is generally not recommended.
    #
    #### Optimization options:
    #
    #   random_seed : int (optional)
    #       Random seed for initializing feature partitions (omega). Each simulation uses 
    #       the sum of the total number of simulations so far and this seed.
    #
    #   perc_score_improve_thresh : float (optional)
    #       Strictly positive threshold specifying the minimum relative improvement in the 
    #       objective value required to continue optimizing with the current delta. If not met, 
    #       moves to the next (smaller) delta.
    #
    #
    #### Parallelization:
    #
    #   is_parallel : bool
    #       Whether to parallelize computations during the construction of affinity matrices (W).
    #
    #   block_size : int
    #       Positive integer ≤ N. Defines the number of samples per parallel block when computing W 
    #       (via FP_get_W()). If block_size ≥ N, parallelization is effectively disabled.

    def __init__(self, K=2,  perplexity=10,\
                 delta_steps= 10 ,delta_degrad=2., delta_init=None, delta_list=None,\
                 perc_score_improve_thresh=1e-4, random_seed= 0, \
                 is_parallel=True, block_size=100):

        # Initialize problem parameters
        throw_exception(K, 'K', bigger_than_val=1)
        throw_exception(perplexity, 'perplexity', bigger_than_val=0)

        self.K= K
        self.perplexity= perplexity

        # Initialize optimization parameters
        self.delta_steps = delta_steps
        self.delta_degrad= delta_degrad
        self.delta_init= delta_init
        
        
        if delta_list is not None:
            if delta_list[-1]!=0.:
                raise Exception('The last delta should be 0 and not :'+str(delta_list[-1]))
            self.delta_list= delta_list.copy()
        else:
            self.delta_list= None

        self.random_seed= random_seed
        self.perc_score_improve_thresh= perc_score_improve_thresh

        self.is_parallel= is_parallel
        self.block_size= block_size
        
        # Initialize optimal and counter parameters
        self.best_val= np.inf 
        self.best_W = None
        self.best_omega = None
        self.best_inv_epsilons=None
        self.best_sim= None
        
        self.sims=0

    # Returns the optimal feature partitions found by our algorithm. 
    # For each partition, returns the indices of features assigned to it.
    #
    # Output:
    #   feature_partitions_indexes : list of K numpy.ndarrays
    #       A list of K arrays, where each array contains the indices of the features 
    #       assigned to the corresponding partition (i.e., where omega[k, d] > 0.5).
    def get_feature_partitions(self):
        return [ np.where(self.best_omega[i]>0.5)[0] for i in range(self.K)]

    # The function returns the optimal parameters found by our algorithm.
    #
    # Output:
    #   omegas : numpy.ndarray of shape (K, D)
    #       Binary matrix (values 0 or 1), where K is the number of feature partitions 
    #       and D is the number of features. A feature is assigned to partition k if 
    #       omegas[k, d] = 1, and 0 otherwise.
    #
    #   W : numpy.ndarray of shape (K, N, N)
    #       Each of the K matrices is a row-stochastic affinity matrix (shape N x N), 
    #       corresponding to one feature partition.
    #
    #   inv_epsilons : numpy.ndarray of shape (K, N)
    #       Entry (k, i) contains the inverse bandwidth used for sample i in partition k.
    #
    def get_optimal_parameters(self):
        return {'omega':self.best_omega.copy() ,'W':self.best_W.copy(), 'inv_epsilons':self.best_inv_epsilons.copy()}


    # Returns additional information related to the convergence of the algorithm.
    #
    # Output:
    #   simulation_id : int
    #       Index of the simulation in which the best parameters were found.
    #
    #   objective_value : float
    #       Objective value associated with the best parameters found during optimization.
    #
    def get_optimization_summary(self):
        return {'simulation_id':self.best_sim, 'objective_value':self.best_val }
        

    # The function fits the model to the data.
    #
    # Input:
    #
    #   x : array-like or tuple
    #       1. If ndarray: shape (N, D), where N is the number of samples and D is the dimension.
    #       2. If tuple: SVD decomposition of the data with structure (U, S, VT):
    #            - U: (N, S), left singular vectors
    #            - S: (S,), singular values
    #            - VT: (S, D), right singular vectors
    #
    #   verbose : bool (optional)
    #       If True, prints progress information during fitting.
    #
    #   max_iters : int
    #       Maximal number of iterations to perform for each value of delta.
    #
    #   simulations : int
    #       Number of independent simulations to run; the best parameters 
    #       across these runs will be retained.
    #
    #   init_omega : numpy.ndarray of shape (K, D), optional
    #       Non-negative initialization for the feature partitions, where each row 
    #       sums to one. If provided, disables random initialization.
    #       NOTE: If used, recommend setting simulations=1, as repeated simulations 
    #       with the same initialization yield identical results.
    def fit(self,x,verbose= False, max_iters=1000, simulations=1, init_omega=None ):
        
        # Validate the input data and extracts its dimension.
        is_x_svd= isinstance(x,tuple)
        if is_x_svd:
            verify_svd(x)
            N= len(x[0])
            self.dim = np.shape(x[2])[1]
            
        else:
            if len(np.shape(x))!=2:
                raise Execption('The input data should be etiher a tuple containing the SVD decomposition, or a samples-by-coordinates matrix.')
            N= len(x)
            self.dim = np.shape(x)[1]
        
                   
        # Define the regularization coefficient list.
        if self.delta_list is None:
            if self.delta_init is None:
                delta_init = get_delta_init(x,self.K, self.perplexity, self.is_parallel, self.block_size)
            else:
                delta_init= self.delta_init.copy()
            
            delta_list= np.append(  delta_init/ (self.delta_degrad**np.arange(self.delta_steps)),0.)
        else:
            delta_list= self.delta_list.copy()
            
        for sim in range(simulations):

            if verbose:
                print('Begin sim '+str(sim))
                print('Initializing parameters')
            self.sims+=1
            
            is_converged=False 
            curr_val = np.inf

            # Initialize the feature partition weights
            if init_omega is None:
                if self.random_seed is not None:
                    np.random.seed(self.random_seed+self.sims)
        
                omega = np.random.RandomState().rand(self.K,self.dim)
                omega = omega/ np.sum(omega,axis=0,keepdims=True)
                
            else:
                omega = init_omega.copy()


            is_converged=False
            for delta in delta_list:
                
                # If all the values of omega converged approximately to zero or one then get to the last step where delta=0
                if delta>0 and np.all(np.max(omega,axis=0)>1-1e-6):
                    if verbose:
                        print('All features are nearly fully assigned to a single partition. '+\
                              'Therefore, we proceed to the final step where delta = 0.')
                    delta=0.
                    is_converged = True
                
                last_val=np.inf

                if verbose:
                        print("Assigning regularization coefficient (delta): " + str(delta))

                for j in range(max_iters):
                    W,inv_epsilons = FP_get_W(x, omega,self.perplexity, is_parallel=self.is_parallel, block_size=self.block_size)

                    omega= FP_get_omega(x, W, delta )


                    curr_val = FP_get_val(x, W,omega, self.perplexity, delta)


                    
                    # Update the optimal parameters only when there is no regularization within the optimization problem
                    if delta==0 and curr_val< self.best_val:
                        self.best_W= W.copy()
                        self.best_inv_epsilons= inv_epsilons.copy()
                        self.best_omega= omega.copy()
                        self.best_val= curr_val.copy()
                        self.best_sim= self.sims+0

                        if verbose:
                            print('Found new parameters. Simulation number:'+str(sim)+', with value: '+str(self.best_val))


                    # If the objective value does not decrease sufficiently, proceed to the next delta.
                    if j>0 and self.perc_score_improve_thresh is not None:
                        if last_val- curr_val <= np.abs(last_val)*self.perc_score_improve_thresh:
                            break
                    last_val= curr_val.copy()
                
                if is_converged:
                    break
            
