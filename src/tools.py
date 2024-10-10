import math
import numpy as np
import copy
#import matplotlib.pylab as plt


# Give root equations one variables Newton method
def rootNtest(x0,fName,diagnostic=0):
    def difF(x0,h):
        t=2*h
        if isinstance(fName(x0+h), complex):
            lf=fName(x0)
            t=h
        else:
            lf=fName(x0+h)
        if isinstance(fName(x0-h), complex):
            rf=fName(x0)
            t=h
        else:
            rf=fName(x0-h)
        out=(lf-rf)/t
        if out==0:
            out=float('nan')
        return out

    BESTACC=-14
    N=300
    hi=1e-12
    xi=x0
    xout=[xi]
    hout=[hi]

    for i in range(N):
        hi=fName(xi)/difF(xi,abs(hi))
        xi=xi-hi
        hout.append(hi)
        xout.append(xi)
        if abs(hi)<1e-13 or isinstance(hi, complex)or math.isnan(xi):
            hout.pop()
            break

    hout=[math.log(abs(itt))/math.log(10) if itt!=0 else BESTACC for itt in hout]
    if  abs(fName(xi))>1e-12:
        xout[-1]=float('nan')
    if diagnostic==0:
        return [xout[-1],hout[-1]]
    else:
        return [xout,hout]


def rootN(x0,fName,diagnostic=0):
    def difF(x0):
        h=1e-6
        if isinstance(fName(x0+h), complex):
            lf=fName(x0)
            t=h
        else:
            lf=fName(x0+h)
        if isinstance(fName(x0-h), complex):
            rf=fName(x0)
            t=h
        else:
            rf=fName(x0-h)
        out=(lf-rf)/2/h
        if out==0:
            out=float('nan')
        return out

    def mainfunct(x0):
        BESTACC=-14
        N=300
        xi=x0
        xout=[xi]
        hout=[math.nan]

        for i in range(N):
            hi=fName(xi)/difF(xi)
            xi=xi-hi
            hout.append(hi)
            xout.append(xi)
            if abs(hi)<1e-12 or isinstance(hi, complex)or math.isnan(xi):
                break

        hout=[math.log(abs(itt))/math.log(10) if itt!=0 else BESTACC for itt in hout]
        if abs(fName(xi))>1e-10:
            xout[-1]=float('nan')
            print('Bad fit!!')
        return [xout,hout]

    if isinstance(x0,list):
        xinit=np.linspace(x0[0],x0[1],50)
        for itt in xinit:
            xout,hout=mainfunct(itt)
            if xout[-1]>=x0[0] and xout[-1]<=x0[1] and not(math.isnan(xout[-1])):
                break
            else:
                xout[-1]=math.nan
    else:
        xout,hout=mainfunct(x0)

    if diagnostic==0:
        return [xout[-1],hout[-1]]
    else:
        return [xout,hout]

def complex_rootN(x0,fName,diagnostic=0):
    def difF(x0):
        h=1e-6
        lf=fName(x0+h)
        rf=fName(x0-h)
        out=(lf-rf)/2/h
        if out==0:
            out=float('nan')
        return out

    N=300
    xi=x0
    xout=[xi]
    hout=[math.nan]

    for _ in range(N):
        hi=fName(xi)/difF(xi)
        xi=xi-hi
        hout.append(abs(hi))
        xout.append(xi)

        if abs(hi) < 1e-12:
            break

    if abs(fName(xi))>1e-10:
        xout[-1]=float('nan')
        print('Bad fit!!')

    if diagnostic==0:
        return [xout[-1],hout[-1]]
    else:
        return [xout,hout]



#  Give root equations many variables Newton method
def rootNSystem(x0,fName,diagnostic=0):
    H=1e-6
    Narg=len(x0)
    def fA(x0):
        wsum=np.zeros((Narg,Narg))
        for itt in range(Narg):
            for j in range(Narg):
                xim=copy.deepcopy(x0)
                xip=copy.deepcopy(x0)
                xim[j]=xim[j]-abs(H)
                xip[j]=xip[j]+abs(H)
                wsum[itt,j]=(fName(xip)[itt]-fName(xim)[itt])/2/abs(H)
        return wsum
    def fB(x):
        B=np.zeros(Narg)
        for itt in range(Narg):
            B[itt]=-fName(x)[itt]
        return B

    BESTACC=-14
    N=50
    hi=H*np.ones(Narg)
    xi=np.array(x0,float)
    hout=[math.nan]
    xout=[xi]
    for i in range(N):
        A=fA(xi)
        B=fB(xi)
        hi=np.linalg.solve(A,B)
        xi=xi+hi
        xout.append(xi)
        sch=max(abs(hi))
        hout.append(sch)
        if sch<1e-13:
            break

    hout=[math.log(itt)/math.log(10) if itt!=0 else BESTACC for itt in hout]
    y=fName(xi)
    scy=0 #SKO
    for itt in y:
        scy+=itt**2
    scy=(scy/Narg)**0.5
    if scy>1e-12:
        xi=float('nan')

    if diagnostic==0:
        return [xi,hout[-1]]
    else:
        return [xout,hout]


# Integral simpson method
def intSims(lim,fName,nInt):
    ld=lim[0]
    lu=lim[1]
    nump=2*nInt+1
    h=float(lu-ld)/(nump-1)
    sum2=0
    sum4=0
    for i in range(1,nump-1,2):
        sum4+=fName(ld+i*h)
        sum2+=fName(ld+i*h+h)
    return h/3*(fName(ld)-fName(lu)+4*sum4+2*sum2)


# Integral Monte-Carlo method
def intMonte(lim,fName,n):
    def rand(lim):
        n_int=8
        subsets=np.arange(0,n+1,n/n_int)
        steps=n/n_int
        u=np.zeros(n)
        for ii in range(n_int):
            start=int(subsets[ii])
            end=int(subsets[ii+1])
            u[start:end]=np.random.uniform(low=ii/n_int,high=(ii+1)/n_int,size=end-start)
        np.random.shuffle(u)
        return lim[0]+(lim[1]-lim[0])*u
    out=0
    D=1
    u=np.zeros((len(lim),n))
    for ii in range(len(lim)):
        u[ii,:]=rand(lim[ii])
        D*=lim[ii][1]-lim[ii][0]
    for kk in range(n):
        out+=fName(u[:,kk])
    return out/n*D


# Integral recurrent method
def intEitken(lim,fName,nPow=14,diagnostic=0):
    def midInt(N,fName):
        h=(lim[1]-lim[0])/N
        x=np.linspace(lim[0]+h/2,lim[1]-h/2,N)
        u=fName(x)
        return sum(u)*h

    def reccur(Ui,ni):
        #ni номер уточнения ni = 1,2,3,4,5...nrec
        Un=np.zeros(nPow)
        q=np.zeros(nPow)
        R=np.zeros(nPow)
        dU=np.zeros(nPow)
        dU[2*ni-2]=Ui[2*ni-1]-Ui[2*ni-2]
        for i in range(2*ni,nPow):
            dU[i-1]=Ui[i]-Ui[i-1]
            q[i]=dU[i-2]/dU[i-1]
            R[i]=dU[i-1]/(q[i]-1)
            if math.isnan(R[i]) or math.isinf(R[i]):
                R[i]=0
            Un[i]=Ui[i]+R[i]
        return np.log(abs(q))/np.log(r),np.log(abs(R))/np.log(10),Un

    r=2 # Удваиваем чило промежутков
    nrec=round(nPow/2)-1  # Число реккурентных уточнений (на каждое уточнение тратится 2 сетки всего сеток nPow)
    nInt=r**np.arange(nPow) # массив числа промежутков 1,2,4,8,..2**(nPow-1) (размер nPow)
    Ui=np.zeros(nPow)  #массив для значений интегралов вычесленных для каждого nInt
    n=np.log10(r)*np.arange(nPow)
    for i in range(nPow):
        Ui[i]=midInt(nInt[i],fName)
    qi,Ri,Ui=reccur(Ui,1)
    U=[Ui]; R=[Ri]; q=[qi]

    if Ri[2]<Ri[-1] and Ri[-1]>0:
        print('*********************')
        print('int does not converge')
        print('*********************')
        return math.nan,math.nan
    for i in range(1, nrec):
        qi,Ri,Ui =reccur(U[i-1],i+1)
        U.append(Ui)
        R.append(Ri)
        q.append(qi)

    '''
    if diagnostic==1:
        fig2=plt.figure('Логарифм ошибки')
        axes2=fig2.add_axes([0.1,0.1,0.8,0.8])
        plt.xlabel('lg(N)')
        plt.grid()
        fig3=plt.figure('Порядок точности')
        axes3=fig3.add_axes([0.1,0.1,0.8,0.8])
        plt.xlabel('lg(N)')
        plt.grid()
        for j in range(nrec):
            axes2.plot(n,R[j],n,R[j],'o')
            axes3.plot(n,q[j],n,q[j],'o')
        plt.show(block=False)
    '''

    return U[-1][-1],R[-1][-1]


def accInt(lim,fName,metName,nPow=6):
    nInt=2**np.arange(nPow)
    delt=np.zeros(nPow)
    deltt=np.ones(nPow)
    ma=np.ones(nPow)
    inti=np.zeros(nPow)
    for i in range(nPow):
        inti[i]=metName(lim,nInt[i],fName)
        delt[i]=abs(inti[i]-inti[i-1])/(2**ma[i]-1)
        deltt[i]=abs(inti[i]-inti[i-1])
        ma[i]=1/np.log(2)*np.log(deltt[i-1]/deltt[i])
    return nInt,np.log(delt)/np.log(10),ma,inti


def intSimscum(lim,nInt,fName,nOut=100):
    ld=lim[0]
    lu=lim[1]
    nump=2*nInt+1;  # Число точек интегрирования
    h=float(lu-ld)/(nump-1)
    intC=np.zeros(nInt+1)
    for i in range(1,nump-1,2):
        intC[int((i+1)/2)]=intC[int((i+1)/2)-1]+h/3*(fName(ld+i*h-h)+4*fName(ld+i*h)+fName(ld+i*h+h))
    return intC


def intcum(lim,f_name,nOut=100,nInt=14):
    diapH=abs(lim[1]-lim[0])/(nOut-1)
    intC=np.zeros(nOut)
    for i in range(nOut):
        intC[i],_=intEitken([lim[0],diapH*i],f_name,nInt)
    return intC


# 2d Integral Simple  method
def int2d(lim,n_points,f_name):
    nx=n_points
    ny=n_points
    limx=lim[0]
    limy=lim[1]
    hy=abs(limy[1]-limy[0])/ny
    hx= abs(limx[1]-limx[0])/nx

    xi=np.linspace(limx[0],limx[1],nx+1)
    yi = np.linspace(limy[0], limy[1], ny+1)

    int=0
    for ii in xi[0:-1]:
        for jj in yi[0:-1]:
            int+=f_name(ii+hx/2,jj+hy/2)
    return hx*hy*int


# Euler method
def ode_euler(init_cond, lim, r_funct, n_int=1000, out_err = False):
    def main_funct(N):
        t=np.linspace(lim[0],lim[1],N+1)
        dt=(lim[1]-lim[0])/N
        y=np.zeros(N+1)
        y[0]=init_cond[0]
        for i in range(N):
            ytmp=y[i]+r_funct[0](t[i],y[i])*dt/2
            y[i+1]=y[i]+r_funct[0](t[i]+dt/2,ytmp)*dt
        return t,y

    _, y1=main_funct(n_int)
    _, y2=main_funct(n_int*2)
    t3,y3=main_funct(n_int*4)
    y3=y3[0::4]
    t3=t3[0::4]
    d1=y2[0::2]-y1
    d2=y3-y2[0::2]
    q=d1/d2
    R=d2/(q-1)

    ind=np.where(np.isnan(R))[0]
    R[ind]=0
    q[ind]='inf'

    y3=y3+R
    newR=[]
    for itt in R:
        if math.isnan(itt):
            newR.append(0)
        else:
            newR.append(abs(itt))
    err=max(newR)

    if out_err:
        return t3,y3,np.log(err)/np.log(10)
    else:
        return t3,y3


# CROS method
def ode_cros(init_cond, lim, r_funct, n_int=200, out_err = False):
    def der_fun(ii, t, y):
        h = 1e-5
        return (r_funct(t[ii-1], y[ii-1] * (1 + h)) - r_funct(t[ii-1], y[ii-1] * (1 - h))) / (2 * h * y[ii-1])

    def main_funct(N):
        t = np.linspace(lim[0], lim[1], N+1)
        dt = (lim[1] - lim[0]) / N

        y = np.zeros(N + 1)
        y[0] = init_cond

        a = np.complex128('1+j') / 2.0

        for ii in range(1, N+1):
            w = r_funct(t[ii-1] + dt / 2.0, y[ii-1]) / (1 - a * der_fun(ii, t, y) * dt)
            y[ii] = y[ii-1] + np.real(w) * dt
        
        return t, y

    _, y1 = main_funct(n_int)
    _, y2 = main_funct(n_int*2)
    t3, y3 = main_funct(n_int*4)

    y3 = y3[0::4]
    t3 = t3[0::4]

    d1 = y2[0::2] - y1
    d2 = y3 - y2[0::2]
    q = d1 / d2

    R = d2 / (q-1)
    R[np.where(np.isnan(R))[0]] = 0

    y3 = y3 + R

    if out_err:
        return t3, y3, max([0 if math.isnan(itt) else abs(itt) for itt in abs(R)])
    else: 
        return t3, y3


# Runge-Kutta method
def ode45(init_cond, lim, r_funct, n_int=1000, out_err=False):
    n_var = len(init_cond)

    def main_funct(N):
        t=np.linspace(lim[0],lim[1], N+1)
        dt = (lim[1]-lim[0])/N
        y = -math.inf*np.ones((N+1, n_var))

        y[0] = init_cond

        for i in range(N):
            arg1 = y[i,:]
            f1 = r_funct(arg1,t[i])

            arg2 = y[i,:] + dt/2 * np.array(f1)
            f2 = r_funct(arg2,t[i] + dt/2)

            arg3 = y[i,:] + dt/2 * np.array(f2)
            f3 = r_funct(arg3,t[i] + dt/2)

            arg4 = y[i,:] + dt * np.array(f3)
            f4 = r_funct(arg4, t[i] + dt)

            y[i+1,:] = y[i,:] + dt/6 * (np.array(f1) + 2*np.array(f2) + 2*np.array(f3) +np.array(f4))

        return t,y
    
    _,y1 = main_funct(n_int)
    _,y2 = main_funct(n_int*2)
    t3,y3 = main_funct(n_int*4)

    y3 = y3[0::4,:]
    t3 = t3[0::4]

    d1 = y2[0::2,:] - y1
    d2 = y3 - y2[0::2,:]
    q = d1 / d2

    R = d2 / (q-1)
    R[np.where(np.isnan(R[:,0]))[0]] = 0

    y3 = y3 + R

    if out_err:
        return t3, y3, max([0 if math.isnan(itt) else abs(itt) for itt in abs(R[:,0])])
    else:
        return t3, y3


# Gauss method
class ExGaussNoSquare(Exception):
    pass


class ExGaussBnoA(Exception):
    pass

def lingauss(A,B,ntp=0):
    def changemax(M,S,crcl):
        ind=crcl+abs(M[crcl:,crcl]).argmax()
        tmpM=M[crcl,crcl:].copy()
        tmpS=S[crcl].copy()

        S[crcl]=S[ind]
        M[crcl,crcl:]=M[ind,crcl:]
        S[ind]=tmpS
        M[ind,crcl:]=tmpM

    try:
        A=np.array(A,'float')
        B=np.array(B,'float')
        n=len(A[0,:])

        if A.ndim!=2 or n!=len(A[:,0]):
            raise ExGaussNoSquare()
        if len(B)!=len(A):
            raise ExGaussBnoA()

        X=np.zeros(n)
        N=0

        if ntp==1:
            A0=A.copy()
            B0=B.copy()

        for imcl in range(n):
            changemax(A,B,imcl)
            for irw in range(imcl+1,n):
                if A[irw,imcl]!=0:
                    alph=-A[irw,imcl]/A[imcl,imcl]
                    B[irw]+=B[imcl]*alph
                    for icl in range(imcl,n):
                        A[irw,icl]+=A[imcl,icl]*alph

        for imcl in sorted(range(n),reverse=True):
            tmpb=B[imcl]
            for itt in range(imcl,n):
                tmpb-=A[imcl,itt]*X[itt]
            X[imcl]=tmpb/A[imcl,imcl]

        if ntp==1:
            for imrw in range(n):
                tmpn=-B0[imrw]
                for imcl in range(n):
                    tmpn+=A0[imrw,imcl]*X[imcl]
                N+=abs(tmpn)
            N=N/n
    except ExGaussNoSquare:
        print('Матрица не квадратная')
        if ntp==1:
            return None,None
        else:
            return None
    except ExGaussBnoA:
        print('Размер матрицы B не соответствует A')
        if ntp==1:
            return None,None
        else:
            return None
    else:
        if ntp==1:
            return X,N
        else:
            return X


# Gauss method, без выбора главного элемента
def lingaussNCng(A,B,ntp=0):
    try:
        A=np.array(A,'float');
        B=np.array(B,'float')
        n=len(A[0,:])

        if A.ndim!=2 or n!=len(A[:,0]):
            raise ExGaussNoSquare()
        if len(B)!=len(A):
            raise ExGaussBnoA()

        X=np.zeros(n)
        N=0

        if ntp==1:
            A0=A.copy()
            B0=B.copy()

        for imcl in range(n):
            for irw in range(imcl+1,n):
                if A[irw,imcl]!=0:
                    alph=-A[irw,imcl]/A[imcl,imcl]
                    B[irw]+=B[imcl]*alph
                    for icl in range(imcl,n):
                        A[irw,icl]+=A[imcl,icl]*alph

        for imcl in sorted(range(n),reverse=True):
            tmpb=B[imcl]
            for itt in range(imcl,n):
                tmpb-=A[imcl,itt]*X[itt]
            X[imcl]=tmpb/A[imcl,imcl]

        if ntp==1:
            for imrw in range(n):
                tmpn=-B0[imrw]
                for imcl in range(n):
                    tmpn+=A0[imrw,imcl]*X[imcl]
                N+=abs(tmpn)
            N=N/n
    except ExGaussNoSquare:
        print('Матрица не квадратная')
        if ntp==1:
            return None,None
        else:
            return None
    except ExGaussBnoA:
        print('Размер матрицы B не соответствует A')
        if ntp==1:
            return None,None
        else:
            return None
    else:
        if ntp==1:
            return X,N
        else:
            return X


# Tridiagonal matrix algoritm
#
# b - main diagonal
# a - lower diaganal
# c - upper diaganal
def tridiag_alg(a,b,c,r):
    n=len(b)
    alf=np.zeros(n)
    beta=np.zeros(n)
    out=np.zeros(n)
    alf[0]=-c[0]/b[0]
    beta[0]=r[0]/b[0]
    for ii in range(1,n-1):
        alf[ii]=-c[ii]/(b[ii]+alf[ii-1]*a[ii])
        beta[ii]=(r[ii]-a[ii]*beta[ii-1])/(b[ii]+alf[ii-1]*a[ii])
    beta[-1]=(r[-1]-a[-1]*beta[-2])/(b[-1]+alf[-2]*a[-1])
    out[-1]=beta[-1]
    for ii in reversed(range(n-1)):
        out[ii]=alf[ii]*out[ii+1]+beta[ii]
    return out


# Polinom approximate/interpolate
class ExNoOneDim(Exception):
    pass


class ExLenxiNoyi(Exception):
    pass


def polyapr(xi,yi,m):
    try:
        xi=np.array(xi)
        yi=np.array(yi)
        A=[]
        B=[]
        c=[]
        npnt=len(xi)
        if xi.size!=yi.size:
            raise ExLenxiNoyi()

        if npnt>m:
            for jj in range(m*2):
                summac=0
                summab=0
                for itt in range(npnt):
                    summac+=xi[itt]**jj
                    if jj<m:
                        summab+=(xi[itt]**jj)*yi[itt]
                c.append(summac)
                if jj<m:
                    B.append(summab)

            for jj in range(m):
                line=[]
                for itt in range(jj,m+jj):
                    line.append(c[itt])
                A.append(line)
        else:
            m=npnt
            for itt in range(m):
                line=[]
                for jj in range(m):
                    line.append(xi[itt]**jj)
                A.append(line)
            B=yi

        A=np.array(A)
        B=np.array(B)
        X=lingauss(A,B)

    except ExLenxiNoyi:
        print('Размер xi и yi не совпадает')
        return None
    except ExGaussBnoA:
        return None
    else:
        return X


# Polinom calculate
def polyval(p,x):
    try:
        p=np.array(p)
        if p.ndim!=1:
            raise ExNoOneDim()

        n=len(p)
        out=[]
        for itt in x:
            line=0
            for jj in range(n):
                line+=p[jj]*itt**jj
            out.append(line)
    except TypeError:
        line=0
        for jj in range(n):
            line+=p[jj]*x**jj
        return line
    except ExNoOneDim:
        print('Матрица коэффициентов полинома не вектор')
        return None
    else:
        return np.array(out)


def polyder(p,nder=1):
    try:
        p=np.array(p)
        if p.ndim!=1:
            raise ExNoOneDim()

        for itt in range(p.size):
            p[itt]*=itt

    except ExNoOneDim:
        print('Матрица коэффициентов полинома не вектор')
        return None
    else:
        return p[1:]



# Spline
class Spline():
    def __init__(self,xi,yi,y2sh=[0,0]):
        ind=np.isnan(xi)
        xi=xi[~ind]
        yi=yi[~ind]
        indy=np.isnan(yi)
        xi=xi[~indy]
        yi=yi[~indy]
        self.spline=[]
        self.nder=0
        self.n=len(xi)
        self.snull={'a':float('nan'),'b':0,'x':0,'c':0,'d':0}
        if xi==[] or yi==[]:
            print('Warning xi is empty')
        try:
            if self.n!=len(yi) or self.n==1:
                raise ExLenxiNoyi()

            for itt in range(self.n):
                self.spline.append({'a':yi[itt],'x':xi[itt]})
            self.spline[0]['c']=y2sh[0]/2
            self.spline[-1]['c']=y2sh[-1]/2

            A=np.zeros(self.n-1)
            B=np.zeros(self.n-1)
            B[0]=self.spline[0]['c']
            for itt in range(1,self.n-1):
                hi=xi[itt]-xi[itt-1]
                hi1=xi[itt+1]-xi[itt]
                a=hi
                b=2.*(hi+hi1)
                d=3.*((yi[itt+1]-yi[itt])/hi1-(yi[itt]-yi[itt-1])/hi)
                z=b+a*A[itt-1]
                A[itt]=-a/z
                B[itt]=(d-a*B[itt-1])/z

            for itt in sorted(range(1,self.n-1),reverse=True):
                self.spline[itt]['c']=B[itt]+A[itt]*self.spline[itt+1]['c']

            for itt in range(self.n-1):
                if math.isnan(xi[itt]) or math.isnan(yi[itt]):
                    print('Warning element[',itt,'] is NaN')
                hi1=xi[itt+1]-xi[itt]
                self.spline[itt]['d']=(self.spline[itt+1]['c']-self.spline[itt]['c'])/hi1/3
                self.spline[itt]['b']=(yi[itt+1]-yi[itt])/hi1-hi1/3*(2*self.spline[itt]['c']+self.spline[itt+1]['c'])

        except ExLenxiNoyi:
            print('Размер xi и yi не совпадает')

    def calcspln(self,xin):
        try:
            arr=[]
            for xi in xin:
                if math.isnan(xi):
                    arr.append(float('nan'))
                else:
                    if xi<self.spline[0]['x']:
                        #s=self.spline[0]
                        s=self.snull
                    elif xi>self.spline[-1]['x']:
                        #s=self.spline[-2]
                        s = self.snull
                    else:
                        lbnd=0
                        ubnd=self.n
                        while ubnd-lbnd>1:
                            cmpval=(ubnd+lbnd)//2
                            if xi<=self.spline[cmpval]['x']:
                                ubnd=cmpval
                            else:
                                lbnd=cmpval
                        s=self.spline[lbnd]
                    arr.append(s['a']+s['b']*(xi-s['x'])+(s['c']+s['d']*(xi-s['x']))*(xi-s['x'])*(xi-s['x']))
            return arr
        except TypeError:
            arr=0
            if math.isnan(xin):
                return float('nan')
            if xin<self.spline[0]['x']:
                #s=self.spline[0]
                s = self.snull
            elif xin>self.spline[-1]['x']:
                #s=self.spline[-2]
                s = self.snull
            else:
                lbnd=0
                ubnd=self.n
                while ubnd-lbnd>1:
                    cmpval=(ubnd+lbnd)//2
                    if xin<=self.spline[cmpval]['x']:
                        ubnd=cmpval
                    else:
                        lbnd=cmpval
                s=self.spline[lbnd]
            arr=s['a']+s['b']*(xin-s['x'])+(s['c']+s['d']*(xin-s['x']))*(xin-s['x'])*(xin-s['x'])
            return arr

    def splder(self,nder=1):
        if nder+self.nder>3 or nder<1:
            ds=None
        else:
            ds=copy.deepcopy(self)
            for jj in range(nder):
                ds.nder+=1
                for itt in range(ds.n-1):
                    ds.spline[itt]['a']=ds.spline[itt]['b']
                    ds.spline[itt]['b']=2*ds.spline[itt]['c']
                    ds.spline[itt]['c']=3*ds.spline[itt]['d']
                    ds.spline[itt]['d']=0
        return ds


# B - Spline form 
class B_Spline():
    def __init__(self,xi,yi,w=None,npintr=2,pspl=2):

        self.ndat=len(xi)  # Число точек аппрокс.
        try:
            if self.ndat!=len(yi):
                raise ExLenxiNoyi()
            if npintr<=1:  # Число точек на один сплайн
                npintr=2

            self.pspl=pspl  # Порядок сплайна
            self.nder=0  # Число дифференцирований сплайна
            self.c=[]  # Коэффициенты аппроксимации
            self.x=[]  # Узлы стыковки сплайнов
            self.xi=xi  # Узлы аппроксимации
            self.yi=yi  # Узлы аппроксимации
            if w==None:  # Вес узлов
                self.w=np.ones(self.ndat)
            self.nspl=self.ndat//npintr  # Число сплайнов

            for itt in range(self.pspl,-1,-1):
                self.x.append(xi[0]-itt)

            dn=(self.ndat-self.nspl*npintr)//2

            for itt in range(1,self.nspl):
                self.x.append((xi[dn+itt*npintr]+xi[dn+itt*npintr-1])/2.)

            for itt in range(self.pspl+1):
                self.x.append(xi[-1]+itt)

            A=np.zeros([self.nspl+self.pspl,self.nspl+self.pspl])
            B=np.zeros(self.nspl+self.pspl)

            for itt in range(self.nspl+self.pspl):  # Здесь
                B[itt]=self.scalmultY(itt)  # itt - индекс узла стыковки
                for jj in range(self.nspl+self.pspl):
                    A[itt,jj]=self.scalmultB(itt,jj)
            self.c=np.linalg.solve(A,B)
            # self.c=lingauss(A,B)

        except ExLenxiNoyi:
            print('Размер xi и yi не совпадает')

    def scalmultB(self,ispl,jspl):
        out=0
        for itt in range(self.ndat):
            out+=self.calcB(ispl,self.xi[itt])*self.calcB(jspl,self.xi[itt])*self.w[itt]**2
        return out

    def scalmultY(self,ispl):
        out=0
        for itt in range(self.ndat):
            out+=self.calcB(ispl,self.xi[itt])*self.yi[itt]*self.w[itt]**2
        return out

    def calcB(self,ispl,x):
        p=self.pspl
        if x<=self.x[ispl] or x>self.x[ispl+p+1]:
            return 0

        B=np.zeros(p+1)
        for itt in range(p+1):
            if x>self.x[ispl+itt] and x<=self.x[ispl+itt+1]:
                B[itt]=1
        for pi in range(p):
            for itt in range(p-pi):
                xn=self.x[ispl+itt]
                xn1=self.x[ispl+itt+1]
                xnp1=self.x[ispl+pi+itt+1]
                xnp2=self.x[ispl+pi+itt+2]
                B[itt]=(x-xn)*B[itt]/(xnp1-xn)+(xnp2-x)*B[itt+1]/(xnp2-xn1)
        return B[0]

    def calcdB(self,ispl,x):
        p=self.pspl
        if x<=self.x[ispl] or x>self.x[ispl+p+1] or p==0:
            return 0

        dB=np.zeros(p+1)
        B=np.zeros(p+1)
        for itt in range(p+1):
            if x>self.x[ispl+itt] and x<=self.x[ispl+itt+1]:
                B[itt]=1
        for pi in range(p):
            for itt in range(p-pi):
                xn=self.x[ispl+itt]
                xn1=self.x[ispl+itt+1]
                xnp1=self.x[ispl+pi+itt+1]
                xnp2=self.x[ispl+pi+itt+2]

                dB[itt]=((x-xn)*dB[itt]+B[itt])/(xnp1-xn)+((xnp2-x)*dB[itt+1]-B[itt+1])/(xnp2-xn1)
                B[itt]=(x-xn)*B[itt]/(xnp1-xn)+(xnp2-x)*B[itt+1]/(xnp2-xn1)
        return dB[0]

    def calcspl(self,xin):
        try:
            arr=[]
            for itt in xin:
                line=0
                for jj in range(self.nspl+self.pspl):
                    line+=self.c[jj]*self.calcB(jj,itt)
                arr.append(line)
            return np.array(arr)
        except TypeError:
            arr=0
            for jj in range(self.nspl+self.pspl):
                arr+=self.c[jj]*self.calcB(jj,xin)
            return arr

    def calcderspl(self,xin):
        try:
            arr=[]
            for itt in xin:
                line=0
                for jj in range(self.nspl+self.pspl):
                    line+=self.c[jj]*self.calcdB(jj,itt)
                arr.append(line)
            return np.array(arr)
        except TypeError:
            arr=0
            for jj in range(self.nspl+self.pspl):
                arr+=self.c[jj]*self.calcdB(jj,xin)
            return arr

    def getintervals(self):
        ints=self.x[self.pspl:-self.pspl]
        return ints

    def getbasis(self):
        x=np.linspace(self.xi[0],self.xi[-1],1000)
        line=[]
        for itt in range(self.nspl+self.pspl):
            lst=[]
            for jj in x:
                lst.append(self.calcB(itt,jj))
            line.append(lst)
        return line,x

def binary_search(xfind, arr):
    rbnd=len(arr)-1
    lbnd=0
    while rbnd-lbnd>1:
        tmpbnd=(rbnd + lbnd)//2
        if xfind<=arr[tmpbnd]:
            rbnd=tmpbnd
        else:
            lbnd=tmpbnd
    return lbnd