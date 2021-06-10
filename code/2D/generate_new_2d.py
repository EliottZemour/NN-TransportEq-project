import numpy as np
import scipy.io as sio
import random
import tqdm
import matplotlib.pyplot as plt
import new2d_helpers


#%%
CFL = 0.5 # = np.max(a)*N*T/M ==> M = 2*np.max(a)*T*N
T = 10
ctr = [0.5, 0.5]
r = 0.2
N = 100

M = 4*2e-2*T*N      # M = 80
x = np.linspace(0,1,num=int(N+1))
y = np.linspace(0,1,num=int(N+1))


ini = fgrid_circ(x,y,ctr,r)


#%%

coarse = []
fine = []

#%% Circle
    
"""Fixed parameters"""
CFL = 0.5 # = np.max(a)*N*T/M ==> M = 2*np.max(a)*T*N
T = 10
ctr = [0.5, 0.5]
r = 0.2
Ncoarse = 40
Nfine = 100

Mcoarse = 4*2e-2*T*Ncoarse  # M = 32
Mfine = 4*2e-2*T*Nfine      # M = 80

"""Varying parameters"""
ndata = 25 # will have ndata^2 examples
aloop = np.linspace(-2e-2, 2e-2, num=ndata)


for i in tqdm.tqdm(range(len(aloop))):
    for j in range(len(aloop)):
        a = [aloop[i], aloop[j]]
        coarse.append( np.ravel(solve_circ(a, Mcoarse, Ncoarse, T, ctr, r)) )
        fine.append( np.ravel(solve_circ(a, Mfine, Nfine, T, ctr, r)) )
        
#%% Square
    
"""Fixed parameters"""
CFL = 0.5 # = np.max(a)*N*T/M ==> M = 2*np.max(a)*T*N
T = 10
X = np.array([[0.4, 0.6], [0.4,0.6]])
Ncoarse = 40
Nfine = 100

Mcoarse = 4*2e-2*T*Ncoarse  # M = 16
Mfine = 4*2e-2*T*Nfine      # M = 40

"""Varying parameters"""
ndata = 25 # will have ndata^2 examples
aloop = np.linspace(-2e-2, 2e-2, num=ndata)


for i in tqdm.tqdm(range(len(aloop))):
    for j in range(len(aloop)):
        a = [aloop[i], aloop[j]]
        coarse.append( np.ravel(solve_sqr(a, Mcoarse, Ncoarse, T, X)) )
        fine.append( np.ravel(solve_sqr(a, Mfine, Nfine, T, X)) )
           
     
#%%        
outputdir="C:\\Users\\Eliott\\Desktop\\cse-project\\2D\\newdata\\"
np.savetxt(outputdir+'coarse.csv', coarse, delimiter=',')
np.savetxt(outputdir+'fine.csv', fine, delimiter=',')