import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import *
from network.optimizer import *
from network.multi_layer_network import *
from network.trainer import *

# Load the MNIST dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# Do test for overfit
x_train = x_train[:300]
t_train = t_train[:300]

# Generate the dataset
dataset_train = Dataset(x_train, t_train)
dataset_test = Dataset(x_test, t_test)

# The hyperparameters
max_epochs = 100
iters_num = 1000000000
train_size = x_train.shape[0]
batch_size = 100
num_input_node = x_train.shape[1]
num_output_node = t_train.shape[1]
learning_rate = 0.01
hidden_architecture = [100, 100, 100, 100, 100, 100]
activation_type = 'relu'
weight_init_std = 'he'
weight_decay_lambda = 0.0001
use_dropout = False
dropout_ratio = 0.15
use_batchnorm = False
network = MultiLayerNet(num_input_node, hidden_architecture, num_output_node, activation_type, weight_init_std, \
                        weight_decay_lambda, use_dropout, dropout_ratio, use_batchnorm)
optimizer = SGD(lr=learning_rate)
trainer = NetTrainer(dataset_train, dataset_test, network, max_epochs, iters_num, batch_size, optimizer)

# Do training on the network
trainer.train_network()

# Show the results
plt.plot(trainer.train_acc_list, label='train')
plt.plot(trainer.test_acc_list, label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim([0.0, 1.0])
plt.legend()
plt.show()
