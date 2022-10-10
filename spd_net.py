'''
==============================================================
--- Package with Neural network and optimiser for SPD data ---
==============================================================
'''
# Torch for autograd
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
# Basic libraries for maths
import numpy as np
import scipy
# For plotting the mean
import matplotlib.pyplot as plt
# Function for Stiefel Manifold
from geometric.manifolds.stiefel import stiefel_projection, stiefel_retraction



# Basic linear layer for work with spd matrices
class BiMapLayer(nn.Module):
  def __init__(self, in_size, out_size):
    super().__init__()
    Q, _ = torch.linalg.qr(torch.randn((in_size, in_size)) * torch.sqrt(torch.tensor([2 / out_size], dtype=torch.float32)))
    #Q, _ = np.linalg.qr(np.random.rand(in_size, in_size) * np.sqrt(2 / out_size))
    Q = Q[:out_size, :]
    #Q = torch.tensor(np.round(Q, 14)[:,:out_size].astype(np.float32))
    self.weights = nn.Parameter(Q, requires_grad=True)

  def forward(self, x):
    W = self.weights
    return W.mm(x).mm(W.T)
    #return torch.mm(self.weights, torch.mm(x, self.weights))


# ReLU like Non-linearity layer
class ReEigLayer(nn.Module):
    def __init__(self, threshold = 1e-5):
        super().__init__()
        self.threshold = torch.tensor(threshold)
        
    def forward(self, X):
        eigval, U = torch.linalg.eigh(X)
        eigval = torch.maximum(eigval, self.threshold)
        return U.mm(torch.diag(eigval)).mm(U.T)


# Transition from geometric layers to classical ones
class LogEigLayer(nn.Module):
    def forward(self, X):
        eigval, U = torch.linalg.eigh(X)
        eigval = torch.log(eigval)
        return U.mm(torch.diag(eigval)).mm(U.T)


# Flattening for SPD matrices   
class Triu(nn.Module):
    # Amount of values: (n^2 - n) / 2 + n 
    def forward(self, X):
        rows, cols = X.shape[-2], X.shape[-1]
        return X[torch.triu_indices(rows, cols).unbind()]


# Optimizer on Stiefel Manifold
class RiemOpt_custom(Optimizer):
  def __init__(self, parameters, lr = 1e-3):
    defaults = {'lr': lr}
    super().__init__(parameters, defaults)

  def step(self, closure = None):
        loss = None
        if closure is not None:
          loss = closure()

        with torch.no_grad():
          
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0

                for point in group["params"]:
                    grad = point.grad.data
                    if grad is None:
                        continue
                    
                    # Projection 
                    grad_R = stiefel_projection(grad, point)

                    # Step on tangent space
                    step_vector = point - group['lr'] * grad_R

                    # Retraction
                    point_target = stiefel_retraction(step_vector)
                    point.copy_(point_target)
                    point.requires_grad = True
                    
        return loss