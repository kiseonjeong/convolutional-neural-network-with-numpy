import numpy as np

def numerical_diff(self, f, x):
    """
    (function) numerical_diff
    -------------------------
    - Calculate numerical difference

    ------------------------
    Parameter
    ---------
    - f : input function
    - x : input value(s)

    Return
    ------
    - numerical difference value(s)
    """
    # Calculate a numerical difference
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def __numerical_gradient_without_batch(self, f, x):
    # Calculate numerical gradients without batch
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]            
        x[idx] = float(tmp_val) + h    # f(x+h)
        fxh1 = f(x)            
        x[idx] = float(tmp_val) - h    # f(x-h)
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def numerical_gradient(self, f, x):
    """
    (function) numerical_gradient
    -----------------------------
    Calculate numerical gradient

    Parameter
    ---------
    - f : input function
    - x : input value(s)

    Return
    ------
    - numerical gradient value(s)
    """
    # Calculate numerical gradients with batch
    if x.ndim == 1:
        return __numerical_gradient_without_batch(f, x)
    else:
        grad = np.zeros_like(x)
        for idx, _x in enumerate(x):
            grad[idx] = __numerical_gradient_without_batch(f, _x)
        
    return grad


def gradient_descent(self, f, init_x, lr=0.01, step_num=100):
    """
    (function) gradient_descent
    ---------------------------
    Apply gradient descent algorithm

    Parameter
    ---------
    - f : input function
    - x : initial data
    - lr : learning rate (default = 0.01)
    - step_num : maximum of iterations (default = 100)

    Return
    ------
    - gradient descent result
    """
    # Optimize a function using the gradient descent
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x