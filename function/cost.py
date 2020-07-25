import numpy as np

def sum_of_squares_error(y, t):
    """
    (function) sum_of_squares_error
    -------------------------------
    The sum of squares error

    Parameter
    ---------
    - y : output value(s)
    - t : target value(s)

    Return
    ------
    - error value between value(s)
    """
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t, one_hot_enc=True):
    """
    (function) cross_entropy_error
    ------------------------------
    - The cross entropy error

    Parameter
    ---------
    - y : output value(s)
    - t : target value(s)
    one_hot_enc : one hot encoding flag (default = True)

    Return
    ------
    - error value between value(s)
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    if one_hot_enc == True:
        return -np.sum(t * np.log(y + 1e-7)) / batch_size
    else:
        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size