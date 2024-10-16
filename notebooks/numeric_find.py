import copy
import matplotlib.pylab as plt
import numpy as np
import math

A = 8e-2
l = 1e-3

N = 10
lim = [0, 1]

def rootNSystem(x0, fName):
    H = 1e-6
    Narg = len(x0)
    
    def fA(x0):
        wsum = np.zeros((Narg, Narg))

        for j in range(Narg):
            xim = copy.deepcopy(x0)
            xip = copy.deepcopy(x0)
            
            xim[j] += -H
            xip[j] += H

            wsum[:, j] = (fName(xip) - fName(xim))/2/H
        return wsum

    hi = H*np.ones(Narg)
    xi = np.array(x0, float)

    for i in range(100):
        A = fA(xi)
        B = -np.array(fName(xi))
        hi = np.linalg.solve(A, B)

        xi += hi

        if max(abs(hi)) < 1e-13:
            break

    return xi

def solve(xa, y_init):
    N = len(xa) - 1
    d = (xa[-1] - xa[0])/N
    
    def f(y):
        out = np.zeros(N+1)

        i10 = 0
        for i in range(N):
            i10 += (y[i+1]+y[i])/(xa[i] - xa[0] + d/2)**0.5

        out[0] = y[0] - l*d/2*i10

        for i in range(1, N):
            i1 = 0
            for j in range(i, N):
                i1 += (y[j+1] + y[j])/(xa[j] - xa[i] + d/2)**0.5

            i2 = 0
            for j in range(i):
                i2 += y[j+1] + y[j]

            out[i] = y[i]/A - l*d/2/A*i1 + y[0]/y[i] + (y[i+1] - y[i-1])/4/y[i]/y[i]*i2 - 1
        
        i2N = 0
        for i in range(N):
            i2N += y[i+1] + y[i]

        out[-1] = y[-1]/A + y[0]/y[-1] +(3*y[-1] - 4*y[-2] + y[-3])/4/y[-1]/y[-1]*i2N - 1

        return out

    ya = rootNSystem(y_init, f)

    return ya, np.max(abs(f(ya)))



for i in range(100):
    xa_old = np.linspace(lim[0], lim[1], N+1)
    y_init_old = np.random.rand(N+1)+1e-8

    ya, err = solve(xa, y_init)

    y_init_old = ya
    xa_old = xa

    plt.plot(xa, ya, 'b', label="err = %.4f" % (err))
    plt.plot(xa, y_init, 'r')

plt.grid()
plt.legend()
plt.show()
