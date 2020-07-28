import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import *
from network.optimizer import *
from network.deep_conv_net import DeepConvNet
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
network = DeepConvNet(input_dim=(1, 28, 28), 
                    conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                    conv_param_2={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                    conv_param_3={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                    conv_param_4={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                    conv_param_5={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                    conv_param_6={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                    num_hidden_node=50, num_output_node=10)
trainer = NetTrainer(dataset_train, dataset_test, network, max_epochs, mini_batch_size=100, optimizer='Adam', optimizer_param={'lr': 0.001}, eval_sample_num_per_epoch=1000)

# Do training on the network
trainer.train_network()

# Save the training results
network.save_params("deep_conv_net_params.pkl")
print("Save the training results")

# Show the results
plt.plot(trainer.train_acc_list, label='train')
plt.plot(trainer.test_acc_list, label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim([0.0, 1.0])
plt.legend()
plt.show()
