from collections import OrderedDict
import numpy as np
from activation_function import ActFunc
from cost_function import CostFunc
from numerical_gradient import NumGrad
from network_layer import *

# The neural network with multi-layers
class MultiLayerNet:
    # Object initializer
    def __init__(self, num_input_node, num_hidden_node_list, num_output_node, activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        # Initialize parameters
        self.af = ActFunc()    # activation function
        self.cf = CostFunc()    # cost function
        self.gr = NumGrad()    # gradient calculator
        self.num_input_node = num_input_node    # input node information
        self.num_hidden_node_list = num_hidden_node_list    # hidden node information
        self.num_hidden_layer = len(num_hidden_node_list)    # number of hidden layers
        self.num_output_node = num_output_node    # output node information
        self.weight_decay_lambda = weight_decay_lambda    # weight decay lambda parameter

        # Initialize network parameters        
        self.__init_weight(weight_init_std)

        # Create network architecture
        net_layer_var = {'sigmoid': Sigmoid, 'relu': Relu}    # supported network variables
        self.__create_network(net_layer_var, activation)

    # Initialize weights
    def __init_weight(self, weight_init_std):
        net_architecture = [self.num_input_node] + self.num_hidden_node_list + [self.num_output_node]
        self.params = {}    # network parameters
        for idx in range(1, len(net_architecture)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / net_architecture[idx - 1])    # for ReLU
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / net_architecture[idx - 1])    # for sigmoid
            self.params['W' + str(idx)] = scale * np.random.randn(net_architecture[idx - 1], net_architecture[idx])
            self.params['b' + str(idx)] = np.zeros(net_architecture[idx])

    # Create a network
    def __create_network(self, net_layer_var, activation):
        self.layers = OrderedDict()
        for idx in range(1, self.num_hidden_layer + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = net_layer_var[activation]()
        idx = self.num_hidden_layer + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
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
        for idx in range(1, self.num_hidden_layer + 2):
            grads['W' + str(idx)] = self.gr.numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = self.gr.numerical_gradient(loss_W, self.params['b' + str(idx)])

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
        for idx in range(1, self.num_hidden_layer + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads