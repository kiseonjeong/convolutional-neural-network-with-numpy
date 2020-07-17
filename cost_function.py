import numpy as np

# The cost functions for neural network
class CostFunc:
    # Object initializer
    def __init__(self):
        self.delta = 1e-7

    # Mean squared error
    def mean_squared_error(self, y, t):
        return 0.5 * np.sum((y - t) ** 2)

    # Cross entropy error
    def cross_entropy_error(self, y, t, one_hot_enc=True):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        
        batch_size = y.shape[0]
        if one_hot_enc == True:
            return -np.sum(t * np.log(y + self.delta)) / batch_size
        else:
            return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size