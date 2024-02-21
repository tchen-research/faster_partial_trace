import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import clear_output
from .lanczos import *
from .partial_trace_alg import *

def xlogx(x):
    """
    return x log(x) with negative values of x mapped to zero
    """
    
    x_trimmed = x*(x>=0) # set negative values to zero
    
    return x_trimmed*np.log(x_trimmed + 1*(x<=0)) # ensure zero maps to zero
        
def get_ergotropy(Hs,Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r,E0,β):
    """
    Compute the ergotropy
    """

    # define function
    f = lambda x: np.exp(-β*(x-E0))

    # estimate \rho^*(\beta)
    Tr = form_matfunc_ptrace(f,Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r)
    ρ = Tr / np.trace(Tr)

    Λs,Qs = np.linalg.eigh(Hs.A)
    Λρ,Qρ = np.linalg.eigh(ρ)

    return np.trace(Hs@ρ) - np.sum(Λs*Λρ[::-1])
    #np.trace(Hs@(ρ - Qs@(Qρ.T@ρ@Qρ)[::-1,::-1]@Qs.T))

def get_ergotropy_all(Hs,Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r,E0,βs):
    """
    """

    ergotropy_all = []
    for l,β in enumerate(βs):
        ergotropy_all.append(get_ergotropy(Hs,Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r,E0,β))
        
    return np.array(ergotropy_all)
    
def get_ρ_EVs(Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r,E0,β):
    """
    Compute the eigenvalues of the reduce density matrix trb( exp(-βH) )/tr( exp(-βH) ) given the output of fast_partial_trace_quadrature
    """

    # define function
    f = lambda x: np.exp(-β*(x-E0))

    # estimate \rho^*(\beta)
    Tr = form_matfunc_ptrace(f,Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r)

    # get eigenvalues
    try:
        ρ_EVs = np.linalg.eigvalsh(Tr)/np.trace(Tr)
    except: 
        raise RuntimeError('eigenvalues do not converge')
        
    return ρ_EVs

def get_ρ_EVs_all(Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r,E0,βs):
    """
    Compute the eigenvalues of the reduce density matrix trb( exp(-βH) )/tr( exp(-βH) ) given the output of fast_partial_trace_quadrature for a range of βs
    """

    ρ_EVs_all = []
    for l,β in enumerate(βs):
        ρ_EVs_all.append(get_ρ_EVs(Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r,E0,β))
        
    return np.array(ρ_EVs_all)

def get_vN_entropy(Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r,E0,βs):
    """
    Compute von Neumann entropy for a range of βs given the given the output of fast_partial_trace_quadrature
    """

    n_βs = len(βs)
    vN_entropy = np.full(n_βs,np.nan)

    for l,β in enumerate(βs):
        ρ_EVs = get_ρ_EVs(Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r,E0,β)
        vN_entropy[l] = - np.sum(xlogx(ρ_EVs))

    return vN_entropy