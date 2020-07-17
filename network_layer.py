import numpy as np
from activation_function import ActFunc
from cost_function import CostFunc

# The multiply layer
class MulLayer:
    # Object initializer
    def __init__(self):
        self.x = None
        self.y = None

    # Do forward computations
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    # Do backward computations
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

# The addition layer
class AddLayer:
    # Object initializer
    def __init__(self):
        pass

    # Do forward computations
    def forward(self, x, y):
        out = x + y

        return out

    # Do backward computations
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

# The ReLU layer
class Relu:
    # Object initializer
    def __init__(self):
        self.mask = None

    # Do forward computations
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    # Do backward computations
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

# The Sigmoid layer
class Sigmoid:
    # Object initializer
    def __init__(self):
        self.out = None

    # Do forward computations
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    # Do backward computations
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

# The Affine layer
class Affine:
    # Object initializer
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    # Do forward computations
    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    # Do backward computations
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

# The Softmax with loss layer
class SoftmaxWithLoss:
    # Object initializer
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        self.af = ActFunc()
        self.cf = CostFunc()

    # Do forward computations
    def forward(self, x, t):
        self.t = t
        self.y = self.af.softmax(x)
        self.loss = self.cf.cross_entropy_error(self.y, self.t)

        return self.loss

    # Do backward computations
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx