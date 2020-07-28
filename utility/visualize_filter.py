import sys, os
sys.path.append(os.path.abspath('.'))
import numpy as np
import matplotlib.pyplot as plt
from network.simple_conv_net import SimpleConvNet

def filter_show(filters, nx=8):
    """
    (function) filter_show
    -----------------
    - Show the input filters as the images

    Parameter
    ---------
    - filters : input filters
    - nx : number of images on x axis (default : 8)

    Return
    ------
    - none
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

# Initial parameters
network = SimpleConvNet()
filter_show(network.params['W1'])

# Trained parameters
network.load_params()
filter_show(network.params['W1'])