import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import clear_output
from .lanczos import *


def rank_1_partial_trace(x,d_a,d_b):
    """
    Output tr_b( xx^T )

    Input
    -----
    x   : (d_a*d_b, ) ndarray
    d_a : int
    d_b : int

    Output
    ------
    out: (d_a, d_a) ndarray
    """
        
    # form matrix whose columns are x_(i), where x^T = [x_(1)^T, ..., x_(d_a)^T]
    X = np.reshape(x,(d_a,d_b)).T 
    
    return X.T@X # i,j entry is x_(i)^T x_(j)


def fast_partial_trace_quadrature(H,Q,m,d_a,d_b,n_evec=0,reorth=False,termination_cond=None,max_iter=50):
    """
    Get quadrature nodes and weights for estimating tr_b(f(H)) with an approximation of the form
    $$
    \sum_{i=1}^{k} f(\lambda_i) U_i + \frac{1}{m} \sum_{i=1}^{m} R_l^T f(\theta) R_r.
    $$

    Input
    -----
    H : (d_a*d_b,d_a*d_b) matrix-free operator
    Q : (d_a*d_b,k) ndarray
        orthonormal matrix for deflation
    d_a : int
          dimension for partial trace
    d_b : int
          dimension for partial trace compliment
    n_evec : int
             number of eigenvectors in Q (must be first cols)
    reorth : bool
             use reorthogonalization in Lanczos
    termination_cond : callback function
                       determines when to terminate Lanczos
    max_iter : int
               maximum number of Lanczos iterations

    Returns
    -------
    Λk : (k,) ndarray
    Tr_defl : length k list of (d_a,d_a) ndarray
    Θs : length m list of (t,) ndarray
    Tr_rems_l : length m list of (t,d_a) ndarray
    Tr_rems_r : length m list of (t,d_a) ndarray
    
    """
    
    #===============#
    #   deflation   #
    #===============#
    
    # if Q is empty
    if Q is None: 
        k = 0
        Q = np.zeros((d_a*d_b,0))
        Λk = np.zeros((0))
    else:
        k = Q.shape[1]
        if n_evec == k:
            Λk,Vk = np.linalg.eigh(Q.T@(H@Q)) 
        else:
            _,M,R = block_lanczos(H,Q,Q,max_iter,\
                                  reorth=True,\
                                  Q_defl=Q[:,:n_evec],\
                                  termination_cond=termination_cond\
                                 )

            # construct tridiagonal matrix
            T = get_block_tridiag(M,R[1:])
            Θ,S = np.linalg.eigh(T)

            # form factorization
            Vk = R[0].T@S[:k]
            Λk = Θ

    
    # compute partial trace of facortization terms
    Tr_defl = []
    for i in range(k):
        Tr_defl.append(rank_1_partial_trace(Q@Vk[:,i],d_a,d_b))

        
    #===============#
    #   remainder   #
    #===============#

    Θs = []
    Tr_rems_l = []
    Tr_rems_r = []
    for j in range(m):
        
        # form random sampling matrix
        v = np.random.randn(d_b,1)
        v /= np.linalg.norm(v)
        Y = np.kron(np.eye(d_a),v)
    
        if Q is None:
            QQY = np.zeros_like(Y)
            Q_invariant = True
        else:
            QQY = Q@(Q.T@Y)
    
        Z = Y - QQY
        W = Y + QQY

        Y,M,R = block_lanczos(H,Z,W,max_iter,\
                              reorth=reorth,\
                              Q_defl=Q[:,:n_evec],\
                              termination_cond=termination_cond
                             )
    
        # construct tridiagonal matrix
        T = get_block_tridiag(M,R[1:])
        Θ,S = np.linalg.eigh(T)

        # construct factorization
        Yr = S[:d_a].T@R[0]
        if n_evec == k: # due to stability reasons, this is better if Q is invariant subspace
            Yl = Yr
        else:
            Yl = S.T@Y
        
        # add nodes and weights
        Θs.append(Θ)
        Tr_rems_l.append(np.sqrt(d_b)*Yl)
        Tr_rems_r.append(np.sqrt(d_b)*Yr)

    return Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r


def form_matfunc_ptrace(f,Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r):
    """
    Estimate tr_b(f(H)) with an approximation of the form
    $$
    \sum_{i=1}^{k} f(\lambda_i) U_i + \frac{1}{m} \sum_{i=1}^{m} R_l^T f(\theta) R_r.
    $$

    Input
    -----
    f : scalar function
    Λk : (k,) ndarray
    Tr_defl : length k list of (d_a,d_a) ndarray
    Θs : length m list of (t,) ndarray
    Tr_rems_l : length m list of (t,d_a) ndarray
    Tr_rems_r : length m list of (t,d_a) ndarray

    Output
    ------
    Tr : (d_a,d_a) ndarray
    """
    
    m = len(Θs)
    try:
        d_a = np.shape(Tr_rems_l[0])[1]
    except:
        d_a = np.shape(Tr_defl[0])[1]
    
    Tr = np.zeros((d_a,d_a))
    
    # build deflation part
    if isinstance(Λk,np.ndarray):
        for λi,Trxi in zip(Λk,Tr_defl):
            Tr += f(λi)*Trxi
    
    # build remainder part
    for mi in range(m):
        Θ,Yl,Yr = Θs[mi],Tr_rems_l[mi],Tr_rems_r[mi]
        Tr += (1/m)*(Yl.T@np.diag(f(Θ))@Yr)
    Tr = (Tr + Tr.T)/2
    
    return Tr

def form_matfunc_LOO_ptrace(f,Λk,Tr_defl,Θs,Tr_rems_l,Tr_rems_r):
    """
    Build "leave one out" estimates to tr_b(f(H)) from quadrature nodes and weights
    """
    
    d_a = np.shape(Tr_rems_l[0])[1]
    m = len(Θs)
    
    Tr = np.zeros((m,d_a,d_a))
    for j in range(m):
        
        if isinstance(Λk,np.ndarray):
            for λi,Trxi in zip(Λk,Tr_defl):
                Tr[j] += f(λi)*Trxi

        for mi in range(m):
            if mi != j:
                Θ,Yl,Yr = Θs[mi],Tr_rems_l[mi],Tr_rems_r[mi]
                Tr[j] += (1/(m-1))*(Yl.T@np.diag(f(Θ))@Yr)

        Tr[j] = (Tr[j] + Tr[j])/2

    return Tr

def partial_trace(A,d_a,d_b):
    """
    Output tr_b( A ) by explicit computation
    
    Input
    -----
    A    : (d_a*d_b, d_a*d_b) ndarray
    d_a : int
    d_b : int

    Output
    ------
    out: (d_a, d_a) ndarray
    """
    
    T = np.zeros((d_a,d_a),dtype=A.dtype)

    # loop over blocks of A
    for m in range(d_a): 
        for n in range(d_a):
            
            # trace of each block
            T[m,n] = np.trace(A[m*d_b:(m+1)*d_b,n*d_b:(n+1)*d_b]) 
            
    return T