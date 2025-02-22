import matplotlib.pylab as plt
import numpy as np
import sys
import os

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

A=0.01
l=A

print(A, l)

N = 10
lim = [0, 1]
xa = np.linspace(lim[0], lim[1], N+1)

xa_10 = np.linspace(lim[0], lim[1], 10+1)

y_init_10 = [
    
]

ya_out = []
for i in range(len(y_init_10)):
    ya, err = s.solve(
        xa,
        np.interp(xa, xa_10, y_init_10[i]),
        A,
        l
    )
    ya_out.append(ya[-1])
    plt.plot(xa, ya, 'b--', label="err = %.4f" % (err))

delt = 1e-4
for i in range(200):
    xa = np.linspace(lim[0], lim[1], N+1)
    y_init = 0.1*np.random.rand(N+1)+1e-8

    ya_f, err_f = s.solve(xa, y_init, A, l)

    is_exist = False
    for ya_i in ya_out:
        if abs(ya_f[-1] - ya_i) < delt:
            is_exist = True

    if err_f < delt and not is_exist:
        break

with open('y_init.txt', 'w') as f:
    f.write(",\n".join(f"{x:.18e}" for x in y_init))

plt.plot(xa, ya_f, 'r', label="err = %.4f" % (err_f))

plt.grid()
plt.legend()
plt.show()
