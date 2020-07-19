import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from multi_layer_network import MultiLayerNet
from network_layer import *
from network_optimizer import *

# Load the MNIST dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# Do test for overfit
x_train = x_train[:300]
t_train = t_train[:300]

# The hyperparameters
max_epochs = 20
iters_num = 1000000000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
network = MultiLayerNet(num_input_node=784, num_hidden_node_list=[100, 100, 100, 100, 100, 100], num_output_node=10, activation='relu', use_batchnorm=False)
optimizer = SGD(lr=learning_rate)

# Do training on the network
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)
epoch_count = 0
for i in range(iters_num):
    # Get the mini-batch dataset
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Calculate gradients
    #grads = network.numerical_gradient(x_batch, t_batch)
    grads = network.backprop_gradient(x_batch, t_batch)

    # Update the parameters
    optimizer.update(network.params, grads)

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
        epoch_count += 1
        print("{}/{}, train loss, test loss, train acc, test acc | ".format(epoch_count, max_epochs) + str(train_loss) + ", " + str(test_loss) + ", " + str(train_acc) + ", " + str(test_acc))
        if epoch_count >= max_epochs:
            break

# Show the results
plt.plot(train_acc_list, label='train')
plt.plot(test_acc_list, label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim([0.0, 1.0])
plt.legend()
plt.show()
