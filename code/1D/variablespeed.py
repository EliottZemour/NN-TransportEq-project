"""
Created on Thu Mar 18 19:06:46 2021
@author: Eliott

The solve function implements the variable speed (as a function of x)
with a[i-1] in 1D.
Tests and plots in the same file, for u0 = fwall.
"""

import numpy as np
import scipy.io as sio
import random
import tqdm
import matplotlib.pyplot as plt



#%% Useful functions and solver

def fwall(x, X):
    n_wall = X.shape[1]
    res = 0
    for n in range(n_wall):
        res += 0.5*(np.tanh(200*(x-X[0,n])) - np.tanh(100*(x-X[1,n])))
    return res

def a_func(x,a):
    
    # a = [a0, a1, a2, a3, a4] 
    res = 0
    for k in range(len(a)):
        res += a[k]*np.sin((k+1)*np.pi*x)
    return 5e-2 * np.array(res) / np.max(np.abs(res))

def solve_multiple(a, M, N, T, X):
    M = int(M)
    N = int(N)
    tau = T/M   # time step
    h = 1/N   # space step

    U = np.zeros((N+2,))

    x = np.linspace(0,1,num=int(N+2))

    U = fwall(x, X)
    U[0] = 0
    U[N+1] = 0

    apos = np.where(a<0, 0, a)
    aneg = apos - a
    
    for n in range(0,M):
        Uo = np.copy(U)
        for i in range(1,N+1):
            U[i] = Uo[i] - apos[i-1] * tau/h * (Uo[i] - Uo[i-1]) \
                        - aneg[i] * tau/h * (Uo[i] - Uo[i+1])

    # output is of size N+1
    return U[:-1]


#%% Generate dataset with constant u0
    
T = 10
CFL = 0.5
N = 50
x1 = 0.1
x2 = 0.3
X = np.vstack((x1,x2))

"""Varying parameters"""
N_repet = 5000 # number of examples for each N_wall 
coarse = []
fine = []

for k in tqdm.tqdm(range(N_repet)):
    coefs = np.random.rand(4)
    
    xcoarse = np.linspace(0,1,num=int(N+1))
    a = a_func(xcoarse, coefs)
    amax = np.max(a)
    Mcoarse = amax * T * N / CFL
    coarse.append(solve_multiple(a, Mcoarse, N, T, X))
    
    xfine = np.linspace(0,1,num=int(10*N+1))
    a = a_func(xfine, coefs)
    amax = np.max(a)
    Mfine = amax * T * 10 * N / CFL
    fine.append(solve_multiple(a, Mfine, 10*N, T, X))
        
    
    
outputdir="C:\\Users\\Eliott\\Desktop\\cse-project\\data\\"
#np.savetxt(outputdir+'vspeed-coarse.csv', coarse, delimiter=',')
#np.savetxt(outputdir+'vspeed-fine.csv', fine, delimiter=',')


#%% Generate dataset with random u0
    
T = 10
CFL = 0.5
N = 50

"""Varying parameters"""
N_repet = 100 # number of examples for each N_wall 
N_X = 50
coarse = []
fine = []

for k in tqdm.tqdm(range(N_repet)):
    coefs = np.random.rand(4)
    for l in range(N_X):       
        x1 = random.uniform(0.1, 0.45)
        x2 = random.uniform(0.55, 0.8)
        X = np.vstack((x1,x2))
        
        xcoarse = np.linspace(0,1,num=int(N+1))
        a = a_func(xcoarse, coefs)
        amax = np.max(a)
        Mcoarse = amax * T * N / CFL
        coarse.append(solve_multiple(a, Mcoarse, N, T, X))
        
        xfine = np.linspace(0,1,num=int(10*N+1))
        a = a_func(xfine, coefs)
        amax = np.max(a)
        Mfine = amax * T * 10 * N / CFL
        fine.append(solve_multiple(a, Mfine, 10*N, T, X))
        
    
    
outputdir="C:\\Users\\Eliott\\Desktop\\cse-project\\data\\"
np.savetxt(outputdir+'vspeedu0-coarse.csv', coarse, delimiter=',')
np.savetxt(outputdir+'vspeedu0-fine.csv', fine, delimiter=',')







#%% Parameters of the simulation
    
np.random.seed(seed=22)
T = 10
CFL = 0.5

Nfine = 1e3
Ncoarse = 1e2
x = np.linspace(0,1,num=int(Nfine+1))

#a = a_func(x, [1,1/2,1/3,1/4])

a = a_func(x, np.random.rand(4))
amax = np.max(a)
Mfine = amax * T * Nfine / CFL
Mcoarse = amax * T * Ncoarse / CFL

x1 = 0.1
x2 = 0.3
X = np.vstack((x1,x2))


u_ini = fwall(x,X)
u_pred_fine = solve_multiple(a, Mfine, Nfine, T, X)
u_pred_coarse = solve_multiple(a, Mcoarse, Ncoarse, T, X)
#%%
fig = plt.figure()
plt.plot(x,u_ini,color = 'black',label="$u_{0}(x)$, $N=10^3$")
plt.plot(x,u_pred_fine,color = 'red',label="$u_{pred}(x,T)$, $N=10^3$")
xcoarse = np.linspace(0,1,num=int(Ncoarse+1))
plt.plot(xcoarse,u_pred_coarse,color = 'blue',label="$u_{pred}(x,T)$, $N=10^2$")
plt.xlabel('$x$')
plt.ylabel('$u$')
#plt.title('$a(x) = 0.1 + (x-0.5)^2$, $T=4$, CFL=$0.5$')
plt.legend(loc="upper right")
plt.show()

#fig.savefig("1D-diffusion.png", dpi=500)


#%% plot the speed field
N=500
x = np.linspace(0,1,num=int(N+1))
a1 = a_func(x, np.random.rand(4))
a2 = a_func(x, np.random.rand(4))
a3 = a_func(x, np.random.rand(4))
fig = plt.figure()
plt.plot(x,a1)
plt.plot(x,a2)
plt.plot(x,a3)
plt.xlabel('$x$')
plt.ylabel('$a(x)$')
plt.show()

#fig.savefig('speed_a.png', dpi=400)