"""
Created on Thu Mar 18 19:06:46 2021
@author: Eliott
"""

import numpy as np
import random
import tqdm

#%%

def fwall(x, X):
    n_wall = X.shape[1]
    res = 0
    for n in range(n_wall):
        res += 0.5*(np.tanh(100*(x-X[0,n])) - np.tanh(100*(x-X[1,n])))
    return res

def solve_multiple(a, M, N, T, X):
    M = int(M)
    N = int(N)
    tau = T/M   # time step
    h = 1/N   # space step

    U = np.zeros((M+1,N+1))

    x = h * np.arange(0,N+1,1)

    U[0] = fwall(x, X)
    U[:,0] = 0

    for n in range(0,M):
        #icut = int(N * a * (n+1) * tau)
        #for i in range(icut+1,N+1):
        for i in range(1,N+1):
            U[n+1,i] = U[n,i] - a * tau/h * (U[n,i] - U[n,i-1])

    return U[M,1:] # output is of size N



#%% Creating the data for N = 1,2 walls in order to investigate generalization performance (week5).

"""Fixed parameters"""
N = 50
a = 1e-2
T = 10
CFL = 0.5
N = 50
M = 2 * a * T * N 

"""Varying parameters"""
N_repet = 10#2500 # number of examples for each N_wall 
coarse = []
fine = []


for k in tqdm.tqdm(range(N_repet)):
    x1 = random.uniform(0.1, 0.45)
    x2 = random.uniform(0.55, 0.8)
    X = np.vstack((x1,x2))
    print("X_1=",X)
    #coarse.append(solve_multiple(a, M, N, T, X))
    #fine.append(solve_multiple(a, 20*M, 20*N, T, X))
    x11 = random.uniform(0.1, 0.25)
    x21 = random.uniform(x11 + 0.05, 0.45)
    x12 = random.uniform(x21 + 0.05, 0.65)
    x22 = random.uniform(x12 + 0.05, 0.85)
    x1 = np.array([x11, x12])
    x2 = np.array([x21, x22])
    X = np.vstack((x1,x2))
    print("X_2=",X)
    #coarse.append(solve_multiple(a, M, N, T, X))
    #fine.append(solve_multiple(a, 20*M, 20*N, T, X))

outputdir="C:\\Users\\Eliott\\Desktop\\cse-project\\data\\"
np.savetxt(outputdir+'new-xwall1-2.csv', coarse, delimiter=',')
np.savetxt(outputdir+'new-ywall1-2.csv', fine, delimiter=',')
#%% Creating the data for N = 1,2,3,4,5 walls (week4).

"""Fixed parameters"""
N = 50
a = 1e-2
T = 10
CFL = 0.5
N = 50
M = 2 * a * T * N 

"""Varying parameters"""
N_wall = np.round(np.array([3,4,5])) # number of wall
N_repet = 1000 # number of examples for each N_wall 
coarse = []
fine = []


for k in tqdm.tqdm(range(N_repet)):
#    x1 = random.uniform(0.1, 0.4)
#    x2 = random.uniform(0.5, 0.8)
#    X = np.vstack((x1,x2))
#    coarse.append(solve_multiple(a, M, N, T, X))
#    fine.append(solve_multiple(a, 20*M, 20*N, T, X))
    for n in N_wall:
        xmin = 0.1
        xmax = 0.7
        x1 = np.linspace(xmin, xmax, n)
        h = (xmax-xmin)/(n-1)
        x2 = x1 + 0.5 * h * (1 + 0.1 * np.random.rand(n))
        X = np.vstack((x1,x2))
        coarse.append(solve_multiple(a, M, N, T, X))
        fine.append(solve_multiple(a, 20*M, 20*N, T, X))

outputdir="C:\\Users\\Eliott\\Desktop\\cse-project\\data\\"
np.savetxt(outputdir+'new-xwall50-1000.csv', coarse, delimiter=',')
np.savetxt(outputdir+'new-ywall50-1000.csv', fine, delimiter=',')


#%% Example plot
    
"""Fixed parameters"""
ks = 100
a1 = 1
"""Varying parameters""" 
n = 3 # number of wall
xmin = 0.1
xmax = 0.65

x1 = np.linspace(xmin, xmax, n)
h = (xmax-xmin)/(n-1)

x2 = x1 + 0.5 * h * (1 + 0.2 * np.random.rand(n))
X = np.vstack((x1,x2))


import matplotlib.pyplot as plt
#import preferencefig

fig = plt.figure()
x = np.linspace(0,1,1000)
plt.plot(x, fwall(x, X), c='b', label=r'$u_0(x)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$u$')
#plt.title(r'Multiple walls')
plt.legend(loc=1)
plt.show()
#
fig.savefig('multi-wall-ex.png', dpi=500)

#
#a = 1e-2
#T = 10
#CFL = 0.5
#N = 5000
#M = 2 * a * T * N 
#
#sol = solve_multiple(a, M, N, T, X)
#
#fig = plt.figure()
#x = np.linspace(0,1,num=int(N+1))[1:]
#plt.plot(x, fwall(x, X), c='b', label=r'$u_0(x)$')
#plt.plot(x, sol, c='r', label=r'$u(x,T)$')
#
#plt.xlabel(r'$x$')
#plt.ylabel(r'$u(x)$')
#plt.legend(loc=0, facecolor='white', framealpha=1)
#plt.show
#fig.savefig('transport-wall.png', dpi=300)