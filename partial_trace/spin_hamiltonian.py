import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import clear_output

def kron_list(M):
    """
    return kroneker product of list of matrices M1 kron ... kron Mn

    Input
    -----
    M : list of sparse matrices

    Output : 
    N : sparse matrix 
    
    """
    
    if len(M) == 1:
        return M[0]
    else:
        return sp.sparse.kron(M[0],kron_list(M[1:]))


def get_hamiltonian(Jx,Jy,Jz,h,s):
    """
    General Hamiltoniam matrix for XYZ spin system with isotropic magnetic field in z direction
    $$
    H = \sum_{i,j=1}^{N} \left[  J^{x}_{i,j} σ^{x}_i σ^{x}_j 
        +J^{y}_{i,j} σ^{y}_i σ^{y}_j
        +J^{z}_{i,j} σ^{z}_i σ^{z}_j \right] 
        + \frac{h}{2} \sum_{i=1}^{N} \sigma_i^{z}
    $$

    Input
    -----
    Jx : (N,N) ndarray
    Jy : (N,N) ndarray
    Jz : (N,N) ndarray
    h : float
    
    Output
    ------
    H : ((2*s+1)**N,(2*s+1)**N) real sparse matrix
        Hamiltoanian matrix
    """
    
    N = len(Jx)

    M = int(2*s+1)
    Sx = np.zeros((M,M))
    Sy = np.zeros((M,M),dtype='complex')
    Sz = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            Sx[i,j] = ((i==j+1)+(i+1==j))*np.sqrt(s*(s+1)-(s-i)*(s-j))
            Sy[i,j] = ((i+1==j)-(i==j+1))*np.sqrt(s*(s+1)-(s-i)*(s-j))/1j
            Sz[i,j] = (i==j)*(s-i)*2


    out = sp.sparse.coo_matrix((M**N,M**N))

    for j in range(N):
        for i in range(j):

            I1 = sp.sparse.eye(M**(i))
            I2 = sp.sparse.eye(M**(j-i-1))
            I3 = sp.sparse.eye(M**(N-j-1))

            if Jx[i,j] != 0:
                Sxi_Sxj = kron_list([I1,Sx,I2,Sx,I3])
                out += 2 * Jx[i,j] * Sxi_Sxj

            if Jy[i,j] != 0:
                Syi_Syj = kron_list([I1,Sy,I2,Sy,I3])
                out += 2 * Jy[i,j] * np.real(Syi_Syj)

            if Jz[i,j] != 0:
                Szi_Szj = kron_list([I1,Sz,I2,Sz,I3])
                out += 2 * Jz[i,j] * Szi_Szj

        I1 = sp.sparse.eye(M**j)
        I2 = sp.sparse.eye(M**(N-j-1))

        Sxi_Sxi =  kron_list([I1,Sx@Sx,I2])
        Syi_Syi =  kron_list([I1,np.real(Sy@Sy),I2])
        Szi_Szi =  kron_list([I1,Sz@Sz,I2])

        out += Jx[j,j] * Sxi_Sxi
        out += Jy[j,j] * Syi_Syi
        out += Jz[j,j] * Szi_Szi

        Szj = kron_list([I1,Sz,I2])
        out += h * Szj

    return out.tocsr()