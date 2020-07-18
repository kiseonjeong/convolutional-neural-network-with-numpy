import numpy as np

# Test for gate perceptron
class GatePerceptron:
    # Object initializer
    def __init__(self):
        self.w = np.array([0.0, 0.0])
        self.b = 0.0
    
    # Calculate logical AND gate
    def gate_and(self, x1, x2):
        x = np.array([x1, x2])
        self.w = np.array([0.5, 0.5])
        self.b = -0.7
        tmp = np.sum(self.w * x) + self.b
        if tmp <= 0:
            return 0
        else:
            return 1

    # Calculate logical NAND gate
    def gate_nand(self, x1, x2):
        x = np.array([x1, x2])
        self.w = np.array([-0.5, -0.5])
        self.b = 0.7
        tmp = np.sum(self.w * x) + self.b
        if tmp <= 0:
            return 0
        else:
            return 1

    # Calculate logical OR gate
    def gate_or(self, x1, x2):
        x = np.array([x1, x2])
        self.w = np.array([0.5, 0.5])
        self.b = -0.2
        tmp = np.sum(self.w * x) + self.b
        if tmp <= 0:
            return 0
        else:
            return 1

    # Calculate logical XOR gate
    def gate_xor(self, x1, x2):
        s1 = self.NAND(x1, x2)
        s2 = self.OR(x1, x2)
        y = self.AND(s1, s2)
        return y

