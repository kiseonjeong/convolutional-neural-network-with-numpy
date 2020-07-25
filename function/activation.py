import numpy as np

def identity_function(x):
    """
    (function) identity_function
    ----------------------------
    The identity function

    Parameter
    ---------
    - x : input value(s)

    Return
    ------
    - identity function output value(s)
    """
    return x

def step_function(x):
    """
    (function) step_function
    ------------------------
    The step function

    Parameter
    ---------
    - x : input value(s)

    Return
    ------
    - step function output value(s)
    """
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    """
    (function) sigmoid
    ------------------
    The sigmoid function

    Parameter
    ---------
    - x : input value(s)

    Return
    ------
    - sigmoid function output value(s)
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """
    (function) relu
    ---------------
    The ReLU function

    Parameter
    ---------
    - x : input value(s)

    Return
    ------
    - ReLU function output value(s)
    """
    return np.maximum(0, x)

# Softmax function
def softmax(x):
    """
    (function) softmax
    ------------------
    The softmax function

    Parameter
    ---------
    - x : input vector

    Return
    ------
    - softmax function output vector = probability vector
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
