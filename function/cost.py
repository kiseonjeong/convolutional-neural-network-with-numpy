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

def cross_entropy_error(y, t):
    """
    (function) cross_entropy_error
    ------------------------------
    - The cross entropy error

    Parameter
    ---------
    - y : output value(s)
    - t : target value(s)

    Return
    ------
    - error value between value(s)
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size