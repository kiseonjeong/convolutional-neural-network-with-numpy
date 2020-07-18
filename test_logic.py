import numpy as np

# Do test here
a = np.zeros([2, 3])
a[0, 0], a[0, 1], a[0, 2] = 0.123, 1, 2
a[1, 0], a[1, 1], a[1, 2] = 3, -4, 3.14

print(np.max(a, axis=1))