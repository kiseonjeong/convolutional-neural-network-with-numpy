import numpy as np

# The activation functions for neural network
class actFunc:
    # Object initializer
    def __init__(self):
        pass

    # Step function
    def step_function(self, x):
        return np.array(x > 0, dtype=np.int)

    # Sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # ReLU function
    def relu(self, x):
        return np.maximum(0, x)

    # Identity function
    def identity_function(self, x):
        return x

    # Softmax function
    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))