import numpy as np
from numpy.lib import stride_tricks
from activation_function import ActFunc
from cost_function import CostFunc

# The addition layer
class AddLayer:
    """ 
    Addition layer
    """
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

# The multiplication layer
class MulLayer:
    """ 
    Multiplication layer
    """
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

# The sigmoid layer
class Sigmoid:
    """ 
    Sigmoid (Logistic function) layer
    """
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

# The ReLU layer
class Relu:
    """ 
    ReLU layer
    """
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

# The affine layer
class Affine:
    """ 
    Affine layer
    """
    # Object initializer
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    # Do forward computations
    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    # Do backward computations
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)

        return dx

# The softmax with loss layer
class SoftmaxWithLoss:
    """ 
    Softmax with cross entorpy error layer
    """
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
    """ 
    Dropout regularization layer

    Parameters
    ----------
    dropout_ratio : 0.0 (no dropout), 1.0 (all dropout)\n
    """
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
    """ 
    Batch normalization layer

    Parameters
    ----------
    gamma : scale parameter\n
    beta : shift parameter\n
    momentum : moving average parameter\n
    running_mean : moving average result of batch means\n
    running_var : moving average result of batch variances\n
    """
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
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

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
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

# The convolution layer
class Convolution:
    """ 
    Convolution layer

    Parameters
    ----------
    stride : sliding interval\n
    pad : data padding length\n
    """
    # Object initializer
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    # Do forward computations
    def forward(self, x):
        # Calculate output resolution information
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        # Apply the im2col
        col = self.__im2col(x, FH, FW)
        col_W = self.W.reshape(FN, -1).T

        # Calculate forward computations like affine layer
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    # Convert image to column
    def __im2col(self, input_data, filter_h, filter_w):
        # Calculate result resolution information
        N, C, H, W = input_data.shape
        out_h = (H + 2 * self.pad - filter_h) // self.stride + 1
        out_w = (W + 2 * self.pad - filter_w) // self.stride + 1

        # Do padding on the input data
        img = np.pad(input_data, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        # Generate the column data
        for y in range(filter_h):
            y_max = y + self.stride * out_h
            for x in range(filter_w):
                x_max = x + self.stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stride, x:x_max:self.stride]
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

        return col

    # Do backward computations
    def backward(self, dout):
        # Calculate output resolution information
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # Calculate gradients
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.col_W.T)
        dx = self.__col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

    # Convert column to image
    def __col2im(self, col, input_shape, filter_h, filter_w):
        # Calculate result resolution information
        N, C, H, W = input_shape.shape
        out_h = (H + 2 * self.pad - filter_h) // self.stride + 1
        out_w = (W + 2 * self.pad - filter_w) // self.stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

        # Generate the image data
        img = np.zeros((N, C, H + 2 * self.pad + self.stride - 1, W + 2 * self.pad + self.stride - 1))
        for y in range(filter_h):
            y_max = y + self.stride * out_h
            for x in range(filter_w):
                x_max = x + self.stride * out_w
                img[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]

        return img[:, :, self.pad:H + self.pad, self.pad:W + self.pad]
