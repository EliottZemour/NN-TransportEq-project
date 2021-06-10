"""
Created on Wed Jun  9 11:17:01 2021
@author: Eliott
"""

import numpy as np
import scipy.io as sio
import random
import tqdm
import matplotlib.pyplot as plt

#%%

def fgrid_circ(x,y,center,r):
    Nx = len(x)
    Ny = len(y)
    res = np.zeros((Nx,Ny))
    for  i in range(Nx):
        for j in range(Ny):
            res[i,j] = 0.5*(1 + np.tanh(-60*(np.sqrt((x[i]-center[0])**2 + (y[j]-center[1])**2) - r)))
    return res

def fgrid_sqr(x,y,X):
    Nx = len(x)
    Ny = len(y)
    res = np.zeros((Nx,Ny))
    for  i in range(Nx):
        for j in range(Ny):
            res[i,j] = 0.25*(np.tanh(60*(x[i]-X[0,0]))-np.tanh(60*(x[i]-X[0,1]))) * (np.tanh(60*(y[j]-X[1,0]))-np.tanh(60*(y[j]-X[1,1])))
    return res

def solve_circ(a, M, N, T, ctr, r):
    
    M = int(M)
    N = int(N)
    tau = T/M   # time step
    h = 1/N   # space step
    
    U = np.zeros((M+1,N+1,N+1))
    
    x = np.linspace(0,1,num=int(N+1))
    y = np.linspace(0,1,num=int(N+1))
    #a = a_func(x)
    
    U[0,:,:] = fgrid_circ(x, y, ctr, r)
    
    '''
    In this for loop, we take care of the sign of a since it can be positive and negative
    and the scheme must stay "upwind"
    '''
    
    for n in range(0,M):
        for i in range(1,N):
            for j in range(1,N):
                axp = max(a[0], 0)
                axm = max(-a[0], 0)
                ayp = max(a[1], 0)
                aym = max(-a[1], 0)
                
                U[n+1,i,j] = U[n,i,j] - axp * tau/h * (U[n,i,j] - U[n,i-1,j]) + axm * tau/h * (U[n,i+1,j] - U[n,i,j]) \
                             - ayp * tau/h * (U[n,i,j] - U[n,i,j-1]) + aym * tau/h * (U[n,i,j+1] - U[n,i,j])
    
    return U[M,:,:] # output is of size N

def solve_sqr(a, M, N, T, X):
    
    M = int(M)
    N = int(N)
    tau = T/M   # time step
    h = 1/N   # space step
    
    U = np.zeros((M+1,N+1,N+1))
    
    x = np.linspace(0,1,num=int(N+1))
    y = np.linspace(0,1,num=int(N+1))
    #a = a_func(x)
    
    U[0,:,:] = fgrid_sqr(x, y, X)
    
    '''
    In this for loop, we take care of the sign of a since it can be positive and negative
    and the scheme must stay "upwind"
    '''
    
    for n in range(0,M):
        for i in range(1,N):
            for j in range(1,N):
                axp = max(a[0], 0)
                axm = max(-a[0], 0)
                ayp = max(a[1], 0)
                aym = max(-a[1], 0)
                
                U[n+1,i,j] = U[n,i,j] - axp * tau/h * (U[n,i,j] - U[n,i-1,j]) + axm * tau/h * (U[n,i+1,j] - U[n,i,j]) \
                             - ayp * tau/h * (U[n,i,j] - U[n,i,j-1]) + aym * tau/h * (U[n,i,j+1] - U[n,i,j])
    
    return U[M,:,:] # output is of size N
