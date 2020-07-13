import numpy as np
import matplotlib.pylab as plt
from three_layers_network import *

# Create a neural network
thNet = threeLayersNet()
x, t = thNet.get_data()
thNet.init_network()

# Predict responses
accuracy_cnt = 0
for i in range(len(x)):
    y = thNet.predict(x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))