import numpy as np
import matplotlib.pylab as plt
from numpy.testing._private.utils import nulp_diff
from test_neural_network import testNeuralNet
from activation_function import actFunc
from cost_function import costFunc
from gradient import gradCalc

def function_1(x):
    return 0.01 * x**2 + 0.1 * x

def function_2(x):
    return x[0]**2 + x[1]**2

gc = gradCalc()
init_x = np.array([-3.0, 4.0])
print(gc.gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))