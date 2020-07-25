import numpy as np
import matplotlib.pylab as plt
import cv2 as cv
from dataset.mnist import *
from network.optimizer import *
from network.multi_layer_network import *
from network.trainer import *

# Load the MNIST dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

