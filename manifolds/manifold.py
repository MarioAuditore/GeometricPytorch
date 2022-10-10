'''
============================================
Package with general functions for manifolds
============================================
'''
# Torch for autograd
import torch
# For plotting the mean
import matplotlib.pyplot as plt


# general class to store manifold operations
class Manifold():
  def __init__(self, projection_func, retraction_func, distance_func):
    self.projection = projection_func
    self.retraction = retraction_func
    self.distance = distance_func
    
  def project(self, M, S):
      return self.projection(M, S)

  def retract(self, M, S):
      return self.retraction(M, S)
  
  def calculate_distance(self, X, Y):
      return self.distance(X, Y)


# general function to compute distanse for a set of points
def set_loss(X_set, Y, loss, ord = 1, weights = None):
    if weights == None:
      weights = [1 for _ in range(X_set.shape[0])]
    score = 0
    for i, X in enumerate(X_set):
        score += weights[i] * loss(X, Y) ** ord
    return score


# Mean calculation for manifolds with gradient descent
def iterative_manifold_mean(X_set, manifold, lr = 0.1, n_iter = 100, plot_loss_flag = False):
    
    # init mean with random element from set
    Y = X_set[torch.randint(0, X_set.shape[0], (1,))][0].data
    Y.requires_grad = True
    
    if plot_loss_flag:
      plot_loss = []
      prev_loss = torch.zeros(1)
      plato_iter = 0
      plato_reached = False
    
    for i in range(n_iter):
        
        # calculate loss
        loss = set_loss(X_set, Y, manifold.calculate_distance, ord = 2)

        if plot_loss_flag:
          if torch.allclose(loss, prev_loss):
            if not plato_reached:
              plato_iter = i
              plato_reached = True
          else:
            prev_loss = loss
            plato_reached = False
    
        loss.backward()

        # calculate Riemannian gradient
        riem_grad_Y = manifold.project(Y, Y.grad.data)
        
        # update Y
        with torch.no_grad():
            Y_step = Y - lr * riem_grad_Y

            # project new Y on manifold with retraction
            Y = manifold.retract(Y, Y_step)
        
        # check_positive(Y)
        # non-conformist way of zeroing grad
        Y.requires_grad = True
        
        if plot_loss_flag:
          # collect loss for plotting
          plot_loss.append(loss.data)

    print(f"Total loss: {set_loss(X_set, Y, manifold.calculate_distance, ord = 2)} got in {plato_iter} iterations")
    
    if plot_loss_flag:    
      fig, ax = plt.subplots()
      ax.plot(plot_loss)
      ax.set_xlabel("Iteration")
      ax.set_ylabel("Loss")
      plt.show()
    return Y.data