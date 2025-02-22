import matplotlib.pylab as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import solve as s

#
A = 0.01
l = 0.01

N = 10
lim = [0, 1]

xa_old = np.linspace(lim[0], lim[1], N+1)
y_init_old = [
    2.007098016238779287e-02,
9.061167084221398438e-02,
6.302328130992211619e-02,
8.231130267226893782e-02,
7.470241802021924948e-02,
6.275935547423147631e-02,
3.094102123212385397e-02,
4.757493327695207913e-02,
1.357095499720289674e-03,
4.430362309212870775e-03,
6.884828521449883054e-02
]

ya, err = s.solve(xa_old, y_init_old, A, l)
plt.plot(xa_old, ya, 'r', label="err = %.4f" % (err))

for i in range(60):
    xa = np.linspace(lim[0], lim[1], N+i+1)
    y_init = np.interp(xa, xa_old, y_init_old)

    ya, err = s.solve(xa, y_init, A, l)

    y_init_old = ya
    xa_old = xa

with open('y_init.txt', 'w') as f:
    f.write(",\n".join(f"{x:.18e}" for x in ya))

print(ya[0])

plt.plot(xa, ya, 'b', label="err = %.4f" % (err))

plt.grid()
plt.legend()
plt.show()
