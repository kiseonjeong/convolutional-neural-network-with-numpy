import numpy as np
from activation_function import ActFunc
from cost_function import CostFunc

# The multiplication layer
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

# The sigmoid layer
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

# The affine layer
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

# The softmax with loss layer
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

# The dropout layer
class Dropout:
    # Object initializer
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    # Do forward computations
    def forward(self, x, train_flag=True):
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    # Do backward computations
    def backward(self, dout=1):
        return dout * self.mask

# The batch-normalization layer
class BatchNorm:
    # Object initializer
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None    # in case of convolution layer : 4d, in case of fully connected layer : 2d
        self.running_mean = running_mean
        self.running_var = running_var
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    # Do forward computations
    def forward(self, x, train_flag=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flag)

        return out.reshape(*self.input_shape)

    # Do forward computations
    def __forward(self, x, train_flag=True):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flag:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 1e-7)
            xn = xc / std
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum)
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 1e-7)))

        out = self.gamma * xn + self.beta

        return out

    # Do backward computations
    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N - 1)

        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)

        return dx

    # Do backward computations
    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc0) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        self.dgamma = dgamma
        self.debeta = dbeta

        return dx