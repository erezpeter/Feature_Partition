# BSD 3-Clause License
# 
# Copyright (c) 2007-2024 The scikit-learn developers.
# All rights reserved.
#
# This file is part of scikit-learn and has been modified by Erez Peterfreund on 2025-07-21.
# Modifications:
# - Allowed the input distances to be either float32_t or float64_t.
# - Allowed looking at a consecutive subset of samples and their corresponding distances with all the samples.
# - Allowed to input betas as initialization points.
# - Forced the function to return the extracted betas.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np

from libc cimport math
from libc.math cimport INFINITY

####################################### A copy of sklearn.utils._typedefs.pxd ################
# Commonly used types
# These are redefinitions of the ones defined by numpy in
# https://github.com/numpy/numpy/blob/main/numpy/__init__.pxd.
# It will eventually avoid having to always include the numpy headers even when we
# would only use it for the types.
#
# When used to declare variables that will receive values from numpy arrays, it
# should match the dtype of the array. For example, to declare a variable that will
# receive values from a numpy array of dtype np.float64, the type float64_t must be
# used.
#

ctypedef unsigned char uint8_t
ctypedef unsigned int uint32_t
ctypedef unsigned long long uint64_t
#
# intp_t/np.intp should be used to index arrays in a platform dependent way.
# Storing arrays with platform dependent dtypes as attribute on picklable
# objects is not recommended as it requires special care when loading and
# using such datastructures on a host with different bitness. Instead one
# should rather use fixed width integer types such as int32 or uint32 when we know
# that the number of elements to index is not larger to 2 or 4 billions.
ctypedef Py_ssize_t intp_t
ctypedef float float32_t
ctypedef double float64_t
# Sparse matrices indices and indices' pointers arrays must use int32_t over
# intp_t because intp_t is platform dependent.
# When large sparse matrices are supported, indexing must use int64_t.
# See https://github.com/scikit-learn/scikit-learn/issues/23653 which tracks the
# ongoing work to support large sparse matrices.
ctypedef signed char int8_t
ctypedef signed int int32_t
ctypedef signed long long int64_t

#############################################################################################################

cdef float EPSILON_DBL = 1e-8
cdef float PERPLEXITY_TOLERANCE = 1e-5

ctypedef fused my_float:
    float32_t
    float64_t

def _new_binary_search_perplexity(
        my_float[:, :] sqdistances,
        float desired_perplexity,
        long ind_first_index=-1):
    """Binary search for sigmas of conditional Gaussians.


    Parameters
    ----------
    sqdistances : ndarray of shape (n_subset, n_samples), dtype=np.float32 or np.float64
        Pairwise distances between a group of samples and the full set of samples.  
        The group consists of samples that occur one after another in the dataset,  
        starting at index `ind_first_index`.

    desired_perplexity : float
        Desired perplexity (2^entropy) of the conditional Gaussians.
            
    ind_first_index : int
        The index in the full dataset where this group of consecutive samples begins.
    
    Returns
    -------
    P : ndarray of shape (n_subset, n_samples), dtype=np.float64
        Probabilities of conditional Gaussian distributions p_i|j.
    """
    # Maximum number of binary search steps
    cdef long n_steps = 100

    cdef long n_subset = sqdistances.shape[0]
    cdef long n_samples = sqdistances.shape[1]

    # Precisions of conditional Gaussian distributions
    cdef double beta
    cdef double beta_min
    cdef double beta_max

    # Use log scale
    cdef double desired_entropy = math.log(desired_perplexity)
    cdef double entropy_diff

    cdef double entropy
    cdef double sum_Pi
    cdef double sum_disti_Pi
    cdef long i, j, l
        
    cdef float64_t[:, :] P = np.zeros(
        (n_subset, n_samples), dtype=np.float64)

    cdef float64_t[:] new_betas = np.ones((n_subset,), dtype=np.float64)
        
    for i in range(n_subset):
        beta_min = -INFINITY
        beta_max = INFINITY
        
        beta= new_betas[i]

        # Binary search of precision for i-th conditional distribution
        for l in range(n_steps):
            # Compute current entropy and corresponding probabilities
            # computed over all data
            sum_Pi = 0.0
            for j in range(n_samples):
                if (j != i and ind_first_index==-1)  or (ind_first_index+i!= j  and ind_first_index!=-1):
                    P[i, j] = math.exp(-sqdistances[i, j] * beta)
                    sum_Pi += P[i, j]

            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL
            sum_disti_Pi = 0.0

            for j in range(n_samples):
                P[i, j] /= sum_Pi
                sum_disti_Pi += sqdistances[i, j] * P[i, j]

            entropy = math.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy

            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == INFINITY:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -INFINITY:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        new_betas[i] = beta

    return np.asarray(P), np.asarray(new_betas)