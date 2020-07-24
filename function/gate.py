import numpy as np

# Weight and bias for test
W = np.array([0.0, 0.0])
b = 0.0

def gate_and(x1, x2):
    # Calculate logical AND gate
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(W * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def gate_nand(x1, x2):
    # Calculate logical NAND gate
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(W * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def gate_or(x1, x2):
    # Calculate logical OR gate
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(W * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def gate_xor(x1, x2):
    # Calculate logical XOR gate
    s1 = gate_nand(x1, x2)
    s2 = gate_or(x1, x2)
    y = gate_and(s1, s2)
    return y