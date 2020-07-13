import numpy as np
import matplotlib.pylab as plt
from three_layers_network import *

thNet = threeLayersNet()
thNet.init_network()
x = np.array([1.0, 0.5])
y = thNet.forward(x)
print(y)