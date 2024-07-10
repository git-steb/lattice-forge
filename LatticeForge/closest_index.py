import numpy as np
from scipy.linalg import cholesky
from .closest_index import closestIndexC

def closestIndex(R, x=None):
    G2 = R.T
    U = cholesky(np.dot(G2, G2.T), lower=True)
    flipU = np.diag((np.diag(U) >= 0) * 2 - 1)
    G3 = np.dot(U.T, flipU)
    Q = np.dot(np.dot(flipU, np.linalg.inv(G3)), G2)
    H3 = np.linalg.inv(G3)
    
    if x is not None:
        x = x.reshape(1, -1)
        x3 = np.dot(x, Q.T)
        uhat = closestIndexC(H3, x3)
    else:
        uhat = closestIndexC(H3)
        
    return uhat
