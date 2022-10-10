'''
=============================================
Package with functions for Spherical manifold
=============================================
'''
# Torch for autograd
import torch
# Basic libraries for maths
import numpy as np
# Manifold class
from geometric.manifolds import Manifold


class SphereManifold(Manifold):
    def __init__(self, sphere_center):
        self.projection = projection
        self.retraction = retraction
        self.distance = angle_distance
        self.center = sphere_center
    
    def project(self, M, S):
        return self.projection(M, S, self.center)

    def retract(self, M, S):
        return self.retraction(M, S, self.center)
    
    def calculate_distance(self, X, Y):
        return self.distance(X, Y, self.center)



def projection(M, S, sphere_center):
    with torch.no_grad():
        # Get normal vector as radius
        n = (M - sphere_center) / torch.linalg.norm(M - sphere_center)
        # Find projection
        return S - torch.inner(S - M, n) * n 


def retraction(M, S, sphere_center):
    with torch.no_grad():
        # Get center-tangent_point vector
        d = torch.linalg.norm(S - sphere_center)
        # Get Radius
        r = torch.linalg.norm(M - sphere_center)
        # Final answer
        return S - (S - sphere_center) * (1 - r / d)


def angle_distance(A, B, sphere_center):
    # Center the sphere
    X = A - sphere_center
    Y = B - sphere_center
    # Get Radius
    r = torch.linalg.norm(X)
    # Find angle
    angle = torch.inner(X, Y) / (r ** 2)
    # # To prevent from nan
    if torch.allclose(angle, torch.tensor([1], dtype=torch.float)):
        return 0
    else:
        return r * torch.arccos(angle)