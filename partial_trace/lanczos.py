import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import clear_output


def block_lanczos(H,Z,W,max_iter,reorth=False,Q_defl=None,termination_cond=None):
    """
    Lanczos quantities: Y = V^TW and T = tridiag(M,R)

    Input
    -----
    H    : (n,n) matrix-free operator
    Z    : (n,r) ndarray
    W    : (n,s) ndarray
    max_iter : int
    reorth : bool
    Q_delf : bool
    termination_cond : callback function
                       determines when to terminate Lanczos
    Output
    ------
    Y : (r*n_iter,s) ndarray
    M : lenth n_iter list of (r,r) ndarray
    R : lenth n_iter list of (r,r) ndarray
    
    """
    
    n,r = Z.shape
    n,s = W.shape
    Y = np.zeros((r*max_iter,s),dtype=H.dtype)
    

    M = []
    R = []

    V,Ri = np.linalg.qr(Z)
    R.append(Ri)
    
    if reorth:
        V_full = np.zeros((n,r*max_iter),dtype=H.dtype)
        V_full[:,:r] = V
        
    i = 0
    while i<max_iter:

        Y[i*r:(i+1)*r] = V.conj().T@W
        
        V__ = np.copy(V)
        X = H@V - V_@(R[i].conj().T) if i>0 else H@V
        V_ = V__

        M.append(V_.conj().T@X)
        X -= V_@M[i]
        
        if reorth:
            X -= V_full@(V_full.conj().T@X)

        if Q_defl is not None:
            X -= Q_defl@(Q_defl.conj().T@X)
        
        V,Ri = np.linalg.qr(X)
        R.append(Ri)
        
        if i < max_iter-1 and reorth:
            V_full[:,(i+1)*r:(i+2)*r] = V
            
        if termination_cond is None: 
            pass
        else:
            max_iter = termination_cond(M,R,i,max_iter)
        
        i += 1
        
    return Y[:r*i],M,R
    

def get_block_tridiag(M,R):
    """
    Outputs Lanczos tridiagonal matrix corresponding to diagonal blocks M and off-diagonal blocks R
    
    Input
    -----
    M : lenth k list of (r,r) ndarray
    R : lenth k list of (r,r) ndarray
    
    Output
    ------
    T : (r*k,r*k) ndarray
    """

    k = len(M)
    r = len(M[0])
    
    T = np.zeros((k*r,k*r),dtype=M[0].dtype)

    for i in range(k):
        T[i*r:(i+1)*r,i*r:(i+1)*r] = M[i]

    for i in range(k-1):
        T[(i+1)*r:(i+2)*r,i*r:(i+1)*r] = R[i]
        T[i*r:(i+1)*r,(i+1)*r:(i+2)*r] = R[i].conj().T
        
    return T


def get_termination_cond(f,tol,check_freq,verbose=1):
    """
    callback function for use in Lanczos algorithm to decide when to terminate
    """

    def termination_cond(M,R,i,max_iter):

        if i<1 or (i%check_freq) != 0:
            return max_iter

        r = len(R[0])

        d_a = len(M[0])
        
        T0 = get_block_tridiag(M[:-1],R[1:-1])
        T1 = get_block_tridiag(M,R[1:])

        Θ0,S0 = np.linalg.eigh(T0)
        Θ1,S1 = np.linalg.eigh(T1)

        
        fA0 = S0[:d_a]@(f(Θ0)[:,None]*S0[:d_a].T)
        fA1 = S1[:d_a]@(f(Θ1)[:,None]*S1[:d_a].T)

        if np.linalg.norm(fA0-fA1) / np.linalg.norm(fA1) < tol:
            if verbose>=2:
                print(f'terminating at iteration {i} with successive iterates at distance: {np.linalg.norm(fA0-fA1)/np.linalg.norm(fA1)}')
            return i
        else:
            return max_iter
        
    return termination_cond