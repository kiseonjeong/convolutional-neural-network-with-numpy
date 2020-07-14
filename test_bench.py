import numpy as np
import matplotlib.pylab as plt
from two_layer_network import twoLayerNeuralNet

net = twoLayerNeuralNet(num_input_node=784, num_hidden_node=100, num_output_node=10)
x = np.random.rand(100, 784)
y = net.predict(x)
t = np.random.rand(100, 10)
grads = net.numerical_gradient(x, t)
print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)