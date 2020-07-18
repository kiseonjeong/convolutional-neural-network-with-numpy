from collections import OrderedDict
import numpy as np
from activation_function import ActFunc
from cost_function import CostFunc
from numerical_gradient import NumGrad
from network_layer import *

# The neural network with two layers
class TwoLayerNet:
    # Object initializer
    def __init__(self, num_input_node, num_hidden_node, num_output_node, weight_init_std=0.01):
        # Initialize parameters
        self.af = ActFunc()    # activation function
        self.cf = CostFunc()    # cost function
        self.gr = NumGrad()    # gradient calculator
        self.num_input_node = num_input_node    # input node information
        self.num_hidden_node = num_hidden_node    # hidden node information
        self.num_output_node = num_output_node    # output node information

        # Initialize network parameters
        self.params = {}    # network parameters
        self.params['W1'] = weight_init_std * np.random.randn(num_input_node, num_hidden_node)
        self.params['b1'] = np.zeros(num_hidden_node)
        self.params['W2'] = weight_init_std * np.random.randn(num_hidden_node, num_output_node)
        self.params['b2'] = np.zeros(num_output_node)

        # Create network architecture
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
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
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # Calculate numerical gradients
    def numerical_gradient(self, x, t):
        # Do forward computations
        loss_W = lambda W: self.loss(x, t)

        # Calculate gradients
        grads = {}
        grads['W1'] = self.gr.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = self.gr.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = self.gr.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = self.gr.numerical_gradient(loss_W, self.params['b2'])

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
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
