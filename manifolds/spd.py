'''
===========================================================================
Package with functions for manifold of Symmetric Positive Definite matrices
===========================================================================
'''
# Torch for autograd
from cv2 import exp
import torch
# Basic libraries for maths
import numpy as np
# Matrix operations which pytorch doesn't have yet
from geometric.torch_matrix_operators import matrix_log, matrix_frac_pow, custom_matrix_exp
from geometric.manifolds import Manifold


class SPDManifold(Manifold):
    '''
    Class, representing SPD maniolfd
    main operations
    '''
    def __init__(self):
        self.projection = spd_projection
        self.retraction = spd_retraction
        self.distance = affine_invariant_distance


# Tangent projection on spd manifold
def spd_projection(M, S):
    with torch.no_grad():
        return (M.mm(S + S.T)).mm(M) / 2


# Retraction on spd manifold from tangent space for pytorch
def spd_retraction(M, T):
    with torch.no_grad():
        M_half = matrix_frac_pow(M, 0.5)
        M_negative_half = matrix_frac_pow(M, -0.5)
        
        exponent = torch.linalg.matrix_exp(M_negative_half.mm(T).mm(M_negative_half))
            
        return M_half.mm(exponent).mm(M_half)


# Affine-invariant distance
def affine_invariant_distance(X, Y):
    # Fractional power returns complex values
    frac_pow = matrix_frac_pow(X, -0.5)    
    
    # Calculate product
    prod = frac_pow.mm(Y).mm(frac_pow)

    # Take matrix log
    m_log = matrix_log(prod)
    
    # Find the frobenius norm
    result = torch.linalg.norm(m_log , ord='fro')
    return result


def log_euclidean_distance(X, Y):
    sub = matrix_log(X) - matrix_log(Y)
    return torch.linalg.norm(sub, ord='fro')


# Stein divergence - approximation of spd distance
def stein_divergence(X, Y):
    alpha = torch.log(torch.linalg.det((X + Y) / 2))
    beta = torch.log(torch.linalg.det(X.mm(Y))) / 2
    return alpha - beta 


# Stein divergence - approximation of spd distance (det = 0 case)
def stein_divergence_unstable(X, Y):
    
    k1, k2 = 0, 0
    
    # case for near zero determinant
    if torch.linalg.det((X + Y)/ 2) == 0:
        while torch.linalg.det((X + Y) * torch.exp(torch.tensor([k1])) / 2) == 0:
            k1 += 1
    elif torch.isinf(torch.linalg.det((X + Y)/ 2)):
        while torch.isinf(torch.linalg.det((X + Y) * torch.exp(torch.tensor([k1])) / 2)):
            k1 -= 1

    if torch.linalg.det(X.mm(Y)) == 0:
        while torch.linalg.det(X.mm(Y) * torch.exp(torch.tensor([k2]))) == 0:
            k2 += 1
    elif torch.isinf(torch.linalg.det(X.mm(Y))):
        while torch.isinf(torch.linalg.det(X.mm(Y)  * torch.exp(torch.tensor([k2])))):
            k2 -= 1

    alpha = torch.log(torch.linalg.det((X + Y) * torch.exp(torch.tensor([k1])) / 2)) - torch.tensor([X.shape[0] * k1])
    beta = torch.log(torch.linalg.det(X.mm(Y) * torch.exp(torch.tensor([k2])))) / 2 - torch.tensor([X.shape[0] * k2]) / 2

    return alpha - beta 


# from scipy.linalg import expm, fractional_matrix_power
# # Basic version of retraction function (Not used)
# def spd_retraction_numpy(M, T):
#     with torch.no_grad():
#         # Currently not used
#         return fractional_matrix_power(M, 0.5) @ expm(fractional_matrix_power(M, -0.5) @ T @ fractional_matrix_power(M, -0.5)) @ fractional_matrix_power(M, 0.5) 