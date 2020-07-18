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
        self.af = ActFunc()         # activation function
        self.cf = CostFunc()            # cost function
        self.gr = NumGrad()            # gradient calculator
        self.num_input_node = num_input_node            # input node information
        self.num_hidden_node_list = num_hidden_node_list          # hidden node information
        self.num_hidden_layer = len(num_hidden_node_list)           # number of hidden layers
        self.num_output_node = num_output_node          # output node information
        self.weight_decay_lambda = weight_decay_lambda          # weight decay lambda parameter

        # Initialize network parameters
        self.params = {}            # network parameters
        self.__init_weight(weight_init_std)

        # Create network architecture
        net_layer_var = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        self.__create_network(net_layer_var)
        self.lastLayer = SoftmaxWithLoss()

    # Initialize weights
    def __init_weight(self, weight_init_std):
        pass

    # Create a network
    def __create_network(self, net_layer_var):
        pass

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
        

        return grads