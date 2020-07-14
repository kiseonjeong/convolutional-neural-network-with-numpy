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
        c = np.max(x)
        exp_x = np.exp(x - c)
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x

        return y