import numpy as np

class percept:
    def __init__(self):
        self.w = np.array([0.5, 0.5])
        self.b = -0.7
    
    def AND(self, x1, x2):
        x = np.array([x1, x2])
        tmp = np.sum(self.w * x) + self.b
        if tmp <= 0:
            return 0
        else:
            return 1
