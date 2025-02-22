import numpy as np
import matplotlib.pyplot as plt
import copy

sea = 2e-20
si = 1e-20
eps0 = 8.85e-12
fa = 1100
me = 9.1e-31
qe = 1.6e-19
mi = 40*1.66e-27

P = 1e-5
B = 0.2
na = P/300/1.380/1e-23 * 133

A = si/2/sea*eps0*B**2/me/na
l = (sea*si*mi*fa/qe)**0.5*na/B

print(A, l)

def F(nk, na):
    return (1/3+896*l/735)*nk/A + (1/6-308*l/735)*na/A - 1/4 -3/2*nk/(na-nk) + np.log(na/nk)/2/(na-nk)*(2*na*na-2*nk*na+nk*nk)

def F2(nk, na):
    return (1/6 - 144*l/315)*nk/A + (1/3-8*l/21)*na/A - 1/2 + (na*na-na*nk-4*nk*nk)/2/na/(na-nk)

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

def solve(y_init, A, l):
    def f(y):
        return np.array([
            (1/3+896*l/735)*y[0]/A + (1/6-308*l/735)*y[1]/A - 1/4 -3/2*y[0]/(y[1]-y[0]) + np.log(y[1]/y[0])/2/(y[1]-y[0])*(2*y[1]*y[1]-2*y[0]*y[1]+y[0]*y[0]),
            (1/6 - 144*l/315)*y[0]/A + (1/3-8*l/21)*y[1]/A - 1/2 + (y[1]*y[1]-y[1]*y[0]-4*y[0]*y[0])/2/y[1]/(y[1]-y[0])
        ])

    ya = rootNSystem(y_init, f)

    return ya, np.max(abs(f(ya)))


x = np.linspace(0, 1, 4000)
y = np.linspace(0, 1, 4000)
X, Y = np.meshgrid(x, y)

Z = F(X, Y)
Z2 = F2(X, Y)
r, err = solve([0.07,0.1],A,l)

print(r)
plt.contour(X, Y, Z, levels=[0], colors='blue')
plt.contour(X, Y, Z2, levels=[0], colors='red')
plt.gca().set_aspect('equal')
plt.xlabel('nk')
plt.ylabel('na')
plt.grid(True)
plt.show()
