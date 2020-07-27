from collections import OrderedDict
import pickle
import numpy as np
from network.layer import *

class SimpleConvNet:
    """
    (class) SimpleConvNet
    ---------------------
    - The simple convolutional neural network
    - Conv -> ReLU -> Pooling -> Affine -> ReLU -> Affine -> Softmax

    Parameter
    ---------
    - input_dim : input dimension information on the dataset
    - conv_param : convolution layer parameters
    - num_hidden_node : number of hidden node
    - num_output_node : number of output node
    - weight_init_std : weight sigma value (default = 0.01)
    """
    # Object initializer
    def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                num_hidden_node=100, num_output_node=10, weight_init_std=0.01):
        # Initialize parameters
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        num_input_node = input_dim[1]
        conv_output_size = (num_input_node - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # Initialize network parameters
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, num_hidden_node)
        self.params['b2'] = np.zeros(num_hidden_node)
        self.params['W3'] = weight_init_std * np.random.randn(num_hidden_node, num_output_node)
        self.params['b3'] = np.zeros(num_output_node)

        # Create network architecture
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()

    # Predict a response
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # Calculate a loss value
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    # Calculate an accuracy
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        
        return acc / x.shape[0]

    # Calculate numerical gradients
    def numerical_gradient(self, x, t):
        # Do forward computations
        loss_W = lambda W: self.loss(x, t)

        # Calculate gradients
        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    # Calculate gradients using backpropagations
    def backprop_gradient(self, x, t):
        # Do forward computations
        self.loss(x, t)

        # Do backward computations
        dout = 1
        dout = self.lastLayer.backward(dout)        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Save the gradients
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name='params.pkl'):
        # Save parameters
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name='params.pkl'):
        # Load parameters
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.item():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]