import numpy as np
import copy

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

        if max(abs(hi)) < 1e-8:
            break

    return xi

def solve(xa, y_init, A, l):
    N = len(xa) - 1
    d = (xa[-1] - xa[0])/N
    
    def f(y):
        out = np.zeros(N+1)

        intI0 = 0
        for i in range(N):
            intI0 += (y[i+1]+y[i])/(xa[i] - xa[0] + d/2)**0.5
        intI0 *= d/2

        intPN = 0
        for i in range(N):
            intPN += y[i+1] + y[i]
        intPN *= d/2
# уравнение (16) для i=N
        out[0] = y[-1]/A + (3*y[-1]-4*y[-2]+y[-3])/y[-1]/y[-1]*(intPN/2/d) + (1 - y[0]/A + l/A*intI0)*y[0]*y[0]/y[-1]/y[-1]*(3*y[-1]-4*y[-2]+y[-3])/(-3*y[0]+4*y[1]-y[2]) - 1

# уравнение (16) для внутренних узлов
        for i in range(1, N):
            intI = 0
            for j in range(i, N):
                intI += (y[j+1] + y[j])/(xa[j] - xa[i] + d/2)**0.5
            intI *= d/2
            
            intP = 0
            for j in range(i):
                intP += y[j+1] + y[j]
            intP *= d/2

            out[i] = y[i]/A - l/A*intI + (y[i+1]-y[i-1])/y[i]/y[i]*(intP/2/d) + (1 - y[0]/A + l/A*intI0)*y[0]*y[0]/y[i]/y[i]*(y[i+1]-y[i-1])/(-3*y[0]+4*y[1]-y[2]) - 1
# уравнение (17)
        out[-1] = l*intI0 - y[0]
        
        return out

    ya = rootNSystem(y_init, f)

    return ya, np.max(abs(f(ya)))
