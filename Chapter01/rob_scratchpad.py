import numpy as np

x = np.array([  [0,0,1],
                [0,1,1],
                [12,3,1],
                [1,1,1]])

print("\nx:\n{}".format(x))
y = np.array([[0],[1],[1]])

print("\ny:\n{}".format(y))

xdoty = np.dot(x, y)
print("\nxdoty:\n{}".format(xdoty))