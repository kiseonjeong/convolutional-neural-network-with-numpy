import numpy as np
from network_optimizer import *

# The neural network trainer
class NetTrainer:
    # Object initializer
    def __init__(self, x_train, t_train, x_test, t_test, neural_network,
                max_epochs, iters_num, batch_size, optimizer = Adam()):
        # Initialize parameters
        self.dataset_train = (x_train, t_train)
        self.dataset_test = (x_test, t_test)
        self.neural_network = neural_network
        self.max_epochs = max_epochs        
        self.iters_num = iters_num
        self.train_size = x_train.shape[0]
        self.batch_size = batch_size
        self.iter_per_epoch = max(self.train_size / self.batch_size, 1)
        self.optimizer = optimizer
        self.train_loss_list = []
        self.test_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    # Do training on the network
    def train_network(self):
        # Initialize an epoch count value
        epoch_count = 0
        for i in range(self.iters_num):
            # Get the mini-batch dataset
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.dataset_train[0][batch_mask]
            t_batch = self.dataset_train[1][batch_mask]

            # Calculate gradients
            grads = self.neural_network.backprop_gradient(x_batch, t_batch)

            # Update the parameters
            self.optimizer.update(self.neural_network.params, grads)

            # Calculate the loss value
            loss = self.neural_network.loss(x_batch, t_batch)

            # Calculate an accuracy
            if i % self.iter_per_epoch == 0:
                self.train_loss = self.neural_network.loss(self.dataset_train[0], self.dataset_train[1])
                self.test_loss = self.neural_network.loss(self.dataset_test[0], self.dataset_test[1])
                self.train_acc = self.neural_network.accuracy(self.dataset_train[0], self.dataset_train[1])
                self.test_acc = self.neural_network.accuracy(self.dataset_test[0], self.dataset_test[1])
                self.train_loss_list.append(self.train_loss)
                self.test_loss_list.append(self.test_loss)
                self.train_acc_list.append(self.train_acc)
                self.test_acc_list.append(self.test_acc)
                print("{}/{}, train loss, test loss, train acc, test acc | ".format(epoch_count, self.max_epochs) \
                    + str(self.train_loss) + ", " + str(self.test_loss) + ", " + str(self.train_acc) + ", " + str(self.test_acc))
                if epoch_count >= self.max_epochs:
                    break
                epoch_count += 1
