import numpy as np
# x = np.array([1/7 for _ in range(7)])
# print(np.sum(x))

x = [1/7 for _ in range(7)]
print(abs(sum(x) - 1) < 1e-9)
