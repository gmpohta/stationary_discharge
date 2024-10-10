import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.tools as tl
import matplotlib.pylab as plt
import numpy as np
import math

l=0.1
A=1e-2

delta = 1e-6
xrange = np.arange(0, 0.002, delta)
yrange = np.arange(-0.01, 0.01, delta)

X, Y = np.meshgrid(xrange,yrange)

F = (1-2*l)*X + 0.5*(3/2/l*(1-A/X) + (1+A/X) - 3/2*l*math.pi)*Y
F2 = Y**2 + 5/2*(X + 3/8/l*(1 - A/X))*Y - 10/3*l/A*X**3

plt.contour(X, Y, F, [0], colors='blue')
plt.contour(X, Y, F2, [0], colors='red')
plt.grid()
plt.show()
