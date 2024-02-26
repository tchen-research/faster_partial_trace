import numpy as np
import scipy as sp
from IPython.display import clear_output
from .quantum_density import *

def ground_state_function(h,H1,H2,d_a,d_b,tol=1e-15):
    """
    return ground state von Neumann entropy of H1 + h H2
    """
    
    H = H1 + h*H2
    Λk,Qk = sp.sparse.linalg.eigsh(H,k=1,which='SA',tol=tol)
    partial_g = rank_1_partial_trace(Qk,d_a,d_b)
    ρ_EVs = np.linalg.eigvalsh(partial_g)/np.trace(partial_g)
    vN_entropy = - np.sum(xlogx(ρ_EVs))

    return vN_entropy

def ground_state_function_smooth(h,H1,H2,d_a,d_b,tol=1e-15,k=1,β=1e5):
    """
    return ground state von Neumann entropy of H1 + h H2
    """
        
    H = H1 + h*H2
    m = 0
    
    Λk,Qk = sp.sparse.linalg.eigsh(H,k=k,which='SA')
    E0 = Λk[0]
    
    out = fast_partial_trace_quadrature(H,Qk,Λk,m,d_a,d_b)
    
    β = 1/1e-4
    f = lambda x: np.exp(-β*(x-E0))
    
    ρ_EVs = get_ρ_EVs(*out,E0,β)
    
    vN_entropy = -np.sum(xlogx(ρ_EVs))

    return vN_entropy