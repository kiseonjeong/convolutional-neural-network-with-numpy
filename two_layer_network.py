import numpy as np
from activation_function import actFunc
from cost_function import costFunc
from gradient import grad

# The neural network with two layers
class twoLayerNeuralNet:
    # Object initializer
    def __init__(self, num_input_node, num_hidden_node, num_output_node, weight_init_std=0.01):
        self.af = actFunc()         # activation function
        self.cf = costFunc()            # cost function
        self.gr = grad()            # gradient calculator
        self.num_input_node = num_input_node            # input node information
        self.num_hidden_node = num_hidden_node          # hidden node information
        self.num_output_node = num_output_node          # output node information
        self.params = {}            # network parameters
        self.params['W1'] = weight_init_std * np.random.randn(num_input_node, num_hidden_node)
        self.params['b1'] = np.zeros(num_hidden_node)
        self.params['W2'] = weight_init_std * np.random.randn(num_hidden_node, num_output_node)
        self.params['b2'] = np.zeros(num_output_node)

    # Predict a response
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = self.af.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.af.softmax(a2)

        return y

    # Calculate a loss value
    def loss(self, x, t):
        y = self.predict(x)

        return self.cf.cross_entropy_error(y, t)

    # Calculate an accuracy
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # Calculate gradients
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = self.gr.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = self.gr.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = self.gr.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = self.gr.numerical_gradient(loss_W, self.params['b2'])

        return grads
