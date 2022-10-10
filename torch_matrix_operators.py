'''
============================================
Package with math functions for SPD manifold
============================================
'''

import torch


# Implementation of matrix fractional power for pytorch
# https://discuss.pytorch.org/t/raising-a-tensor-to-a-fractional-power/93655/3
def matrix_frac_pow(M, power):
    evals, evecs = torch.linalg.eigh(M)    
    evpow = evals ** power
    return (evecs.mm(torch.diag(evpow)).mm(evecs.T))


# Matrix logarithm based on same idea as above
def matrix_log(M):
    evals, evecs = torch.linalg.eigh(M)
    evlog = torch.log(evals)
    return (evecs.mm(torch.diag(evlog)).mm(evecs.T))


# Matrix exponent
def custom_matrix_exp(M):
    evals, evecs = torch.linalg.eigh(M)
    evlog = torch.exp(evals)
    return (evecs.mm(torch.diag(evlog)).mm(evecs.T))