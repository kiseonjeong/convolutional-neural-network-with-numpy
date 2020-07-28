import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import *
from network.optimizer import *
from network.simple_conv_net import SimpleConvNet
from network.trainer import *

# Load the MNIST dataset
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# Do test for overfit
x_train = x_train[:5000]
t_train = t_train[:5000]

# Generate the dataset
dataset_train = Dataset(x_train, t_train)
dataset_test = Dataset(x_test, t_test)

# The hyperparameters
max_epochs = 20
network = SimpleConvNet(input_dim=(1, 28, 28), conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        num_hidden_node=100, num_output_node=10, weight_init_std=0.01)
trainer = NetTrainer(dataset_train, dataset_test, network, max_epochs, mini_batch_size=100, optimizer='Adam', optimizer_param={'lr': 0.001})

# Do training on the network
trainer.train_network()

# Save the training results
network.save_params("simple_conv_net_params.pkl")
print("Save the training results")

# Show the results
plt.plot(trainer.train_acc_list, label='train')
plt.plot(trainer.test_acc_list, label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim([0.0, 1.0])
plt.legend()
plt.show()
