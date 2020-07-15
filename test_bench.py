import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from two_layer_network import twoLayerNeuralNet
from layer import *
'''
# Load the MNIST dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# The hyperparameters
iters_num = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.2
network = twoLayerNeuralNet(num_input_node=784, num_hidden_node=50, num_output_node=10)

# Do training on the network
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = 10
#iter_per_epoch = max(train_size / batch_size, 1)
for i in range(iters_num):
    # Get the mini-batch dataset
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Calculate gradients
    grad = network.backprop_gradient(x_batch, t_batch)

    # Update the parameters
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # Calculate a loss value
    loss = network.loss(x_batch, t_batch)

    # Calculate an accuracy
    if i % iter_per_epoch == 0:
        train_loss = network.loss(x_train, t_train)
        test_loss = network.loss(x_test, t_test)
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train loss, test loss, train acc, test acc | " + str(train_loss) + ", " + str(test_loss) + ", " + str(train_acc) + ", " + str(test_acc))

# Show the results
plt.plot(train_loss_list)
plt.show()
'''
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = twoLayerNeuralNet(num_input_node=784, num_hidden_node=50, num_output_node=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.backprop_gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))