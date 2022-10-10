'''
===========================================
Package with functions for Stiefel manifold
===========================================
'''
# Torch for autograd
import torch
# Basic libraries for maths
import numpy as np
# Manifold class
from geometric.manifolds import Manifold


class StiefelManifold(Manifold):
    '''
    Class, representing Stiefel maniolfd
    main operations
    '''
    def __init__(self):
        self.projection = stiefel_projection
        self.retraction = stiefel_retraction
        self.distance = stiefel_distance
    
    def retract(self, M, S):
      return self.retraction(M)


def stiefel_projection(vector, point):
  '''
  Projects a vector on a tangent space of a manifold with the point on it.

  Params:
  -------
  vector : torch.tensor
      Vector we nee to project
  point : torch.tensor
      Point on a manifold

  Returns:
  --------
  vector_R : torch.tensor
      Projection of a vector on a tangent space
  '''
  W_hat = vector.mm(point.t()) - 1/2 * point.mm(point.t().mm(vector.mm(point.t())))
  W = W_hat - W_hat.t()
  vector_R = W.mm(point)
  return vector_R


def stiefel_retraction(tangent_vector):
  '''
  Retraction from tangent space to manifold
  
  Params:
  -------
  tangent_vector : torch.tensor
      Vector from tangent space

  Returns:
  : torch.tensor
      loss (distance) value
  '''
  u, _, vh = torch.linalg.svd(tangent_vector , full_matrices=False)  
  return u.mm(vh)


def stiefel_distance(X, Y, base = None):
    '''
    Stiefel distance (geodesic) as 
    Riemannian metric on Stiefel Tangent space

    Params:
    -------
    X, Y : torch.tensor
        Objects from the manifold
    base : torch.tensor
        Point from the manifold for local projectiion

    Returns:
    : torch.tensor
        Point on a manifold
    '''
    if base == None:
        base = Y

    X_T = stiefel_projection(X, base)
    Y_T = stiefel_projection(Y, base)
    return torch.trace(X_T.T.mm(Y_T))