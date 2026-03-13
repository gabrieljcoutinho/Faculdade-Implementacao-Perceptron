import numpy as np

x = np.array([1,2,3])
w = np.array([0.1,0.2,0.3])
b = 0

z = np.dot(w,x) + b

print(z)