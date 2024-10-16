import matplotlib.pylab as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import solve as s

A = 8e-2
l = 1e-3

N = 10
lim = [0, 1]

xa_old = np.linspace(lim[0], lim[1], N+1)
y_init_old = [
    1.510521973124779316e-01,
    3.420328195211031486e-01,
    5.153125828168685363e-01,
    5.097275891635327794e-01,
    5.911460980395073506e-01,
    8.451869502258292366e-01,
    8.818033806365552785e-01,
    4.849358663794973157e-01,
    4.377341772332690728e-01,
    6.901838387776090267e-01,
    2.334851988017859203e-01
]

for i in [N, int(1.8*N), int(2.7*N), int(4.5*N)]:
    xa = np.linspace(lim[0], lim[1], i+1)
    y_init = np.interp(xa, xa_old, y_init_old)

    ya, err = s.solve(xa, y_init, A, l)

    y_init_old = ya
    xa_old = xa

plt.plot(xa, ya, 'b', label="err = %.4f" % (err))

plt.grid()
plt.legend()
plt.show()
