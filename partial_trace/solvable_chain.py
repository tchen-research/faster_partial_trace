import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import clear_output


def get_solvable_density_EVs(h,J,β,N):
    """
    Get eigenvalues of the reduced density matrix for the first two sites of a length N XY spin chain with given values of h, J, and β
    """
    
    k = np.arange(N)+1

    λk = h - 2*J*np.cos(k*np.pi/(N+1))
    Nk = 1/(1+np.exp(β*λk))

    σ1xσ2x = -4/(N+1)*np.sum(np.sin(k*np.pi/(N+1))*np.sin(2*k*np.pi/(N+1))*Nk)
    σ1z = -1+4/(N+1)*np.sum(np.sin(k*np.pi/(N+1))**2*Nk)
    σ2z = -1+4/(N+1)*np.sum(np.sin(2*k*np.pi/(N+1))**2*Nk)
    σ1zσ2z = σ1z*σ2z-σ1xσ2x**2

    δ = np.sqrt(4*σ1xσ2x**2+(σ1z-σ2z)**2)
    p1 = (1 + σ1z + σ2z + σ1zσ2z)/4
    p2 = (1 - δ - σ1zσ2z)/4
    p3 = (1 + δ - σ1zσ2z)/4
    p4 = (1 - σ1z - σ2z + σ1zσ2z)/4
    
    return np.sort(np.array([p1,p2,p3,p4]))


def get_vN_entropy_true(h,J,N,βs):
    """
    get true von Neumann entropy for the first two sites of a length N XY spin chain with given values of h, J, and βs
    """

    n_βs = len(βs)
    ρ_EVs_true = np.zeros((n_βs,4))      
    vN_entropy_true = np.full(n_βs,np.nan)      
    for l,β in enumerate(βs):

        ρ_EVs_true[l] = get_solvable_density_EVs(2*h,2*J,β,N)
        vN_entropy_true[l] = - np.sum(xlogx(ρ_EVs_true[l]))

    return vN_entropy_true