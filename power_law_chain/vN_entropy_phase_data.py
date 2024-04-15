import numpy as np
import scipy as sp
from scipy import io,integrate,sparse,signal

import matplotlib.pyplot as plt

import os,sys
sys.path.insert(0, '..')

from partial_trace import *

if sys.argv[1] == 'inf':
    α = np.inf
else:
    α = float(sys.argv[1])

# α = 15

directory = f'data/chain_{α}'
os.makedirs(directory, exist_ok=False) # if this experiment already exists, you need to move or delete it (so that we do not end up with data generated with different parameters in the same directory)
os.makedirs(f'{directory}/data') # make subdirectory for data filess = 1/2


# set up system Hamiltonian
s = 1/2
M = int(2*s+1)

N = 16
n = M**N

N_s = 2
N_b = N-N_s
d_a = M**N_s
d_b = M**N_b

J_t = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i != j:
            J_t[i,j] = 1/(np.abs(i-j)**float(α))

J = 1

Jx = (J/2)*J_t
Jy = (J/2)*J_t
Jz = 0*J_t

# save data

# save system parameters
np.save(f'{directory}/dimensions.npy',[s,N,N_s],allow_pickle=True)
np.save(f'{directory}/couplings.npy',[Jx,Jy,Jz],allow_pickle=True)

# save Hamiltonian pieces
H1 = get_hamiltonian(Jx,Jy,Jz,0,s)
H2 = get_hamiltonian(0*J_t,0*J_t,0*J_t,1,s)
sp.sparse.save_npz(f'{directory}/H1',H1)
sp.sparse.save_npz(f'{directory}/H2',H2)

h_min = 0
h_max = 2.25

# get ground state discontinuities
iterations = 10
tolerance = 1e-4
turning_points = []
binary_search_recursive(lambda h: ground_state_function(h,H1,H2,d_a,d_b,tol=1e-5), h_min, h_max, iterations, tolerance, turning_points)
turning_points = np.hstack([[h_min],turning_points,[h_max]])
np.save(f'{directory}/turning_points.npy',turning_points)

acc = (h_max-h_min)/2**iterations # accuracy of turning points

total_nodes = 300 # number of nodes

num_nodes,hs = scaled_cheby(turning_points, total_nodes)
np.save(f'{directory}/num_nodes.npy',num_nodes)
hs = np.hstack([turning_points[1:]-acc,turning_points[:-1]+acc,hs]) # add points to the left and right of the boundaries


# number of deflation vectors
k = 25

# number of random samples
m = 5

out_all = []
for hi,h in enumerate(hs):
    
    experiment_name = f'{h}'
    print(f'field strength: {hi} of {len(hs)}, {experiment_name}')
    clear_output(wait=True)

    os.system(f'python vN_entropy_phase_driver.py {h} {k} {m} {directory} {experiment_name}')
