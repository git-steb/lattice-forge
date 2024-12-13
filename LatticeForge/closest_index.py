from .closest_index_cpp import closestIndexC
import numpy as np
from scipy.linalg import cholesky

def is_upper_triangular(R, tol=1e-10):
    """
    Check if a matrix is upper triangular.
    A matrix is upper triangular if all elements below the diagonal are zero.
    """
    n = R.shape[0]
    for i in range(1, n):
        for j in range(i):
            if abs(R[i, j]) > tol:
                return False
    return True

def closestIndex(R, x=None, allnn=True, epsilon=1e-8, debug=False):
    """
    Find the closest lattice point indices to a given point x.
    """
    R = np.asarray(R, dtype=np.float64)
    if x is not None:
        x = np.asarray(x, dtype=np.float64).flatten()
        
    # Check if R is upper-triangular
    is_upper = is_upper_triangular(R)
    
    if is_upper:
        if x is not None:
            uhat = closestIndexC(R, x, allnn, epsilon)
        else:
            uhat = closestIndexC(R, allnn=allnn, epsilon=epsilon)
    else:
        
        G2 = R.T
        gram = np.dot(G2, G2.T)
        
        try:
            U = cholesky(gram, lower=True)
        except np.linalg.LinAlgError:
            U = cholesky(gram + np.eye(gram.shape[0]) * 1e-12, lower=True)
        
        flipU = np.diag(np.sign(np.diag(U)))
        G3 = U.T @ flipU
        H3 = np.linalg.solve(G3, np.eye(G3.shape[0]))
        Q = flipU @ np.linalg.solve(G3, G2)
        
        if x is not None:
            x3 = x @ Q.T
            uhat = closestIndexC(H3, x3, allnn, epsilon)
        else:
            uhat = closestIndexC(H3, allnn=allnn, epsilon=epsilon)
        
        uhat = np.dot(uhat, np.linalg.inv(Q).T)
        uhat = np.rint(uhat)
        
    uhat = uhat.astype(int)
    
    return uhat
