import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from multi_layer_network import MultiLayerNet
from network_trainer import NetTrainer
from network_optimizer import *

# Do test here
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = MultiLayerNet(num_input_node=784, num_hidden_node_list=[100, 100, 100, 100, 100, 100], num_output_node=10, activation='relu', use_batchnorm=False)
max_epochs = 20
iters_num = 1000000000
batch_size = 100
learning_rate = 0.01
optimizer = Adam(learning_rate)
trainer = NetTrainer(x_train, t_train, x_test, t_test, network, max_epochs, iters_num, batch_size, optimizer)