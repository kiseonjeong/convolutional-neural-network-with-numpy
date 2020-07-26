import numpy as np
import matplotlib.pylab as plt
import cv2 as cv
from dataset.mnist import *
from network.optimizer import *
from network.multi_layer_network import *
from network.trainer import *

d = {'sgd': SGD}
print(SGD)
print(SGD())