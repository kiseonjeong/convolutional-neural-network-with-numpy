from collections import namedtuple
import numpy as np
from network.optimizer import *

# The dataset for trainer
Dataset = namedtuple('Dataset', 'x t')

class NetTrainer:
    """
    (class) NetTrainer
    ------------------
    - The network trainer

    Parameter
    ---------
    - dataset_train : train dataset (x, t)
    - dataset_test : test dataset (x, t)
    - neural_network : network architecture
    - max_epochs : number of maximum epochs
    - iters_num : maximum iterations
    - batch_size : size of batch data
    - optimizer : gradient descent optimizer
    """
    # Object initializer
    def __init__(self, dataset_train, dataset_test, neural_network,
                max_epochs, iters_num, batch_size, optimizer = Adam()):
        # Initialize parameters
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.neural_network = neural_network
        self.max_epochs = max_epochs        
        self.iters_num = iters_num
        self.train_size = dataset_train.x.shape[0]
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
            x_batch = self.dataset_train.x[batch_mask]
            t_batch = self.dataset_train.t[batch_mask]

            # Calculate gradients
            grads = self.neural_network.backprop_gradient(x_batch, t_batch)

            # Update the parameters
            self.optimizer.update(self.neural_network.params, grads)

            # Calculate the loss value
            loss = self.neural_network.loss(x_batch, t_batch)

            # Calculate an accuracy
            if i % self.iter_per_epoch == 0:
                self.train_loss = self.neural_network.loss(self.dataset_train.x, self.dataset_train.t)
                self.test_loss = self.neural_network.loss(self.dataset_test.x, self.dataset_test.t)
                self.train_acc = self.neural_network.accuracy(self.dataset_train.x, self.dataset_train.t)
                self.test_acc = self.neural_network.accuracy(self.dataset_test.x, self.dataset_test.t)
                self.train_loss_list.append(self.train_loss)
                self.test_loss_list.append(self.test_loss)
                self.train_acc_list.append(self.train_acc)
                self.test_acc_list.append(self.test_acc)
                print("{}/{}, train loss, test loss, train acc, test acc | ".format(epoch_count, self.max_epochs) \
                    + str(self.train_loss) + ", " + str(self.test_loss) + ", " + str(self.train_acc) + ", " + str(self.test_acc))
                if epoch_count >= self.max_epochs:
                    break
                epoch_count += 1
