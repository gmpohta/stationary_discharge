import matplotlib.pylab as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import solve as s

A = 8e-2
l = 1e-3

N = 40
lim = [0, 1]

xa_old = np.linspace(lim[0], lim[1], N+1)
y_init_old = 0.1*(0.1+xa_old**2)

for i in [N]:
    xa = np.linspace(lim[0], lim[1], i+1)
    y_init = np.interp(xa, xa_old, y_init_old)

    ya, err = s.solve(xa, y_init, A, l)

    y_init_old = ya
    xa_old = xa

    plt.plot(xa, ya, 'b', label="err = %.4f" % (err))
    plt.plot(xa, y_init, 'r')

plt.grid()
plt.legend()
plt.show()
