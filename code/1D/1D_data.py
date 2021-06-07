"""
Dataset generation file for the training of the neural network
Section [3.1] of project report.

Author: Eliott Zemour
"""

import numpy as np
import tqdm

outputdir="C:\\Users\\Eliott\\Desktop\\cse-project\\reportdata\\"

#%% initial functions & schemes

def fstep(x, x1):
    return (1 + np.tanh(100*(x-x1)))/2

def fwall(x, x1, x2):
    return (np.tanh(100*(x-x1)) - np.tanh(100*(x-x2)))/2

def solve_step(a, M, N, T, x1):
  M = int(M)
  N = int(N)
  tau = T/M   # time step
  h = 1/N   # space step

  U = np.zeros((M+1,N+1))

  x = h * np.arange(0,N+1,1)

  U[0] = fstep(x, x1)
  U[:,0] = 0

  for n in range(0,M):
      icut = int(N * a * (n+1) * tau)
      for i in range(icut+1,N+1):
          U[n+1,i] = U[n,i] - a * tau/h * (U[n,i] - U[n,i-1])

  return U[M,1:] # output is of size N

def solve_wall(a, M, N, T, x1, x2):
  M = int(M)
  N = int(N)
  tau = T/M   # time step
  h = 1/N   # space step

  U = np.zeros((M+1,N+1))

  x = h * np.arange(0,N+1,1)

  U[0] = fwall(x, x1, x2)
  U[:,0] = 0

  for n in range(0,M):
      icut = int(N * a * (n+1) * tau)
      for i in range(icut+1,N+1):
          U[n+1,i] = U[n,i] - a * tau/h * (U[n,i] - U[n,i-1])

  return U[M,1:] # output is of size N



#%%

# parameters of the pde
a = 1e-2
T = 10
CFL = 0.5
N = 5e1
M = int(a * T * N / CFL)


# X stores the coarse approx, Y the fine ones
X = []
Y = []

size = 2500
x1 = np.linspace(0.1, 0.7, size)

# step function loop 
for i in tqdm.tqdm(range(len(x1))):
    X.append( solve_step(a, M, N, T, x1[i]) )  
    #Y.append( solve_step(a, 100*M, 100*N, T, x1[i]) )
   
    
size = 50
x1 = np.linspace(0.2, 0.4, size)
x2 = np.linspace(0.6, 0.8, size)

# wall function loop 
for i in tqdm.tqdm(range(len(x1))):
    for j in range(len(x2)):
        X.append( solve_wall(a, M, N, T, x1[i], x2[j]) )  
        #Y.append( solve_wall(a, 100*M, 100*N, T, x1[i], x2[j]) )
    
    
X = np.array(X)   
Y = np.array(Y) 
#%% uncomment to save the data

#np.savetxt(outputdir+'x50.csv', X, delimiter=',')
#np.savetxt(outputdir+'y1000.csv', Y, delimiter=',')


#%% visualize initial condition, input, targeted output of neural network

import matplotlib.pyplot as plt

x1 = 0.4
x2 = 0.6

a = 1e-2
T = 10
CFL = 0.5
N = 5e1
M = int(a * T * N / CFL)

xcoarse = np.linspace(0,1,num=int(N+1))[1:]
xfine = np.linspace(0,1,num=int(20*N+1))[1:]
x = np.linspace(0,1,num=int(20*N+1))

ini = fwall(x, x1, x2)
sol1 = solve_wall(a, M, N, T, x1, x2)
sol2 = solve_wall(a, 20*M, 20*N, T, x1, x2)



fig = plt.figure()
plt.plot(x, ini, color = 'blue', label="$u_0(x)$")
plt.plot(xcoarse, sol1, color = 'black', label="$u(x,T)$, $N=50$")
plt.plot(xfine, sol2, color = 'red', label="$u(x,T)$, $N=1000$")

plt.xlabel('$x$')
plt.ylabel('$u$')
plt.legend()
plt.show()

#fig.savefig("1D-dataset.png", dpi=500)
#fig.savefig('1d.eps', format='eps')