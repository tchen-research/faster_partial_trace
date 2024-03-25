import numpy as np

import os,sys
sys.path.insert(0, '..')

from partial_trace import *

"""
this is the driver file for an experiment of a system H1 + h H2, in which h is varied.
"""


# argumentsN, N_s, h
h = float(sys.argv[1])
k = int(sys.argv[2])
m = int(sys.argv[3])
directory = sys.argv[4]
experiment_name = sys.argv[5]

# load parameters
[s,N,N_s] = np.load(f'{directory}/dimensions.npy',allow_pickle=True)
N = int(N)
N_s = int(N_s)

M = int(2*s+1)

n = M**N

N_b = N-N_s
d_a = M**N_s
d_b = M**N_b

# build Hamiltonian
H1 = sp.sparse.load_npz(f'{directory}/H1.npz')
H2 = sp.sparse.load_npz(f'{directory}/H2.npz')

H = H1 + h*H2

# get low-energy eigenvectors
Λk,Qk = sp.sparse.linalg.eigsh(H,k=k,which='SA')
E0 = Λk[0]

# use guess of the slowest function to converge and build termination condition
β_hard = 1/1e-1
f_hard = lambda x: np.exp(-β_hard*(x-E0))        
termination_cond = get_termination_cond(f_hard,1e-4,5,verbose=1)

# hard maximum iterations if no convergence
max_iter = 40

# get quadrature nodes/weights
out = fast_partial_trace_quadrature(H,Qk,Λk,m,d_a,d_b,max_iter=max_iter,termination_cond=termination_cond)

data = {
    'out':out,
    'h':h,
    'k':k,
    'm':m,
    }

np.save(f'{directory}/data/{experiment_name}',data,allow_pickle=True)


