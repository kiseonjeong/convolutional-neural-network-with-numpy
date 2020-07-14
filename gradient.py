import numpy as np

# The gradient computations for neural network
class grad:
    # Object initializer
    def __init__(self):
        pass

    # Calculate a numerical difference
    def numerical_diff(self, f, x):
        h = 1e-4
        return (f(x + h) - f(x - h)) / (2 * h)

    # Calculate a numerical gradient
    def numerical_gradient(self, f, x):
        h = 1e-4
        grad = np.zeros_like(x)

        for idx in range(x.size):
            tmp_val = x[idx]
            # f(x+h)
            x[idx] = tmp_val + h
            fxh1 = f(x)

            # f(x-h)
            x[idx] = tmp_val - h
            fxh2 = f(x)

            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val

        return grad

    # Optimize a function using the gradient descent
    def gradient_descent(self, f, init_x, lr=0.01, step_num=100):
        x = init_x

        for i in range(step_num):
            grad = self.numerical_gradient(f, x)
            x -= lr * grad

        return x