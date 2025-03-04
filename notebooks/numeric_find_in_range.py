import matplotlib.pylab as plt
import matplotlib
import numpy as np
import sys
import os
matplotlib.use('Qt5Agg')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import solve as s

sea = 2e-20
si = 1e-20
eps0 = 8.85e-12
fa = 1100
me = 9.1e-31
qe = 1.6e-19
mi = 40*1.66e-27

P = 3e-4
B = 0.2
na = P/300/1.380/1e-23 * 133

A = si/2/sea*eps0*B**2/me/na
l = (sea*si*mi*fa/qe)**0.5*na/B
print(A, l)
#A=0.01
#l=0.0001

N = 20
lim = [0, 1]
ya_out = []

delt = 1e-4
for i in range(200):
    xa = np.linspace(lim[0], lim[1], N+1)
    y_init = 1e-2*np.random.rand(N+1)
    print(y_init)
    ya_f, err_f = s.solve(xa, y_init, A, l)
    print(ya_f, err_f)
    print('---------------------')
    if err_f < delt:
        break

with open('y_init.txt', 'w') as f:
    f.write(",\n".join(f"{x:.18e}" for x in y_init))

plt.plot(xa, ya_f*na, 'r', label="err = %.16f" % (err_f))

plt.xlabel('Потенциал')
plt.ylabel('Плотность электронов м^-3')

plt.grid()
plt.legend()
plt.show()
