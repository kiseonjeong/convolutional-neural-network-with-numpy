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
    - mini_batch_size : size of mini batch data
    - optimizer : gradient descent optimizer
    - optimizer_param : parameter for optimizer
    - eval_sample_num_per_epoch : number of sampling data for evaluations
    - verbose : monitoring flag
    """
    # Object initializer
    def __init__(self, dataset_train, dataset_test, neural_network,
                max_epochs, mini_batch_size, optimizer='SGD', optimizer_param={'lr': 0.01},
                eval_sample_num_per_epoch=None, verbose=True):
        # Initialize parameters
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.neural_network = neural_network
        self.max_epochs = max_epochs
        self.train_size = dataset_train.x.shape[0]
        self.mini_batch_size = mini_batch_size
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iters = int(max_epochs * self.iter_per_epoch)
        optimizer_type = {'sgd': SGD, 'momentum': Momentum, "nesterov": Nesterov, \
                        'adagrad': AdaGrad, 'rmsprop': RMSprop, 'adam': Adam}
        self.optimizer = optimizer_type[optimizer.lower()](**optimizer_param)
        self.eval_sample_num_per_epoch = eval_sample_num_per_epoch
        self.verbose = verbose
        self.curr_iter = 0
        self.curr_epoch = 0
        self.train_loss_list = []
        self.test_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    # Do training on the network
    def train_network(self):
        # train on the dataset
        for i in range(self.max_iters):
            self.__train_step()

        # Evaluate network accuracy
        test_acc = self.neural_network.accuracy(self.dataset_test.x, self.dataset_test.t)
        if self.verbose:
            print("The final test accuracy : " + str(test_acc))

    def __train_step(self):
        # Get the mini-batch dataset
        batch_mask = np.random.choice(self.train_size, self.mini_batch_size)
        x_batch = self.dataset_train.x[batch_mask]
        t_batch = self.dataset_train.t[batch_mask]

        # Calculate gradients
        grads = self.neural_network.backprop_gradient(x_batch, t_batch)

        # Update the parameters
        self.optimizer.update(self.neural_network.params, grads)

        # Calculate the loss value
        loss = self.neural_network.loss(x_batch, t_batch)

        # Calculate an accuracy
        if self.curr_iter % self.iter_per_epoch == 0:
            eval_dataset_train = self.dataset_train
            eval_dataset_test = self.dataset_test
            if not self.eval_sample_num_per_epoch is None:
                eval_dataset_train = Dataset(self.dataset_train.x[:t], self.dataset_train.t[:t])
                eval_dataset_test = Dataset(self.dataset_test.x[:t], self.dataset_test.t[:t])
            self.train_loss = self.neural_network.loss(eval_dataset_train.x, eval_dataset_train.t)
            self.test_loss = self.neural_network.loss(eval_dataset_test.x, eval_dataset_test.t)
            self.train_acc = self.neural_network.accuracy(eval_dataset_train.x, eval_dataset_train.t)
            self.test_acc = self.neural_network.accuracy(eval_dataset_test.x, eval_dataset_test.t)
            self.train_loss_list.append(self.train_loss)
            self.test_loss_list.append(self.test_loss)
            self.train_acc_list.append(self.train_acc)
            self.test_acc_list.append(self.test_acc)
            self.curr_epoch += 1
            if self.verbose:
                print("{}/{}, train loss, test loss, train acc, test acc | ".format(self.curr_epoch, self.max_epochs) \
                    + str(self.train_loss) + ", " + str(self.test_loss) + ", " + str(self.train_acc) + ", " + str(self.test_acc))
        self.curr_iter += 1
