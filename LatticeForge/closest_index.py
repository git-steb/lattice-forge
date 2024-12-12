#from .closest_index_cpp import closestIndexC

import numpy as np
from scipy.linalg import cholesky

def sgn1(x):
    return np.sign(x) if x != 0 else 1.0

def selectRows(m, v):
    return m[v]

def closestIndexC(H, x=None, allnn=True, epsilon=-1.0):
    H = np.array(H)
    n = H.shape[0]

    if epsilon == -1.0:
        epsilon = 1e-8 if allnn else 0.0

    bestdist = np.inf
    k = n
    dist = np.zeros((n, 1))

    e = np.zeros_like(H)
    compCP = x is None or len(x) == 0

    if not compCP:
        x = np.array(x)
        e[k-1] = np.dot(x, H)

    u = np.zeros((1, n))
    u[0, k-1] = np.round(e[k-1, k-1])
    uhat = np.zeros((0, H.shape[1]))
    dhat = np.zeros((0, 1))

    y = (e[k-1, k-1] - u[0, k-1]) / H[k-1, k-1]
    step = np.zeros((n, 1))
    step[k-1] = sgn1(y)

    iteration_counter = 0
    max_iterations = 1000  # Safeguard to prevent infinite loops

    while True:
        iteration_counter += 1
        if iteration_counter > max_iterations:
            print("Warning: Maximum iterations reached, potential infinite loop detected.")
            break

        newdist = dist[k-1] + y * y
        print(f"Considering point u: {u}, newdist: {newdist}, bestdist: {bestdist}, epsilon: {epsilon}")

        if bestdist == np.inf or newdist < bestdist + epsilon:
            if k != 1:
                dist[k-1] = newdist
                e[k-2, :k-1] = e[k-1, :k-1] - y * H[k-1, :k-1]
                k -= 1
                ekk = e[k-1, k-1]
                u[0, k-1] = np.round(ekk)
                y = (ekk - u[0, k-1]) / H[k-1, k-1]
                step[k-1] = sgn1(y) if y != 0 else 1.0  # Ensure step is never zero
                print(f"Updated u in Case A: {u}, y: {y}, step: {step[k-1]}")
            else:
                if not compCP or (newdist != 0):
                    if allnn:
                        uhat = np.concatenate([uhat, u], axis=0)
                        newdist_reshaped = np.array([[newdist]]) if np.ndim(newdist) == 0 else np.array(newdist).reshape(1, 1)
                        print(f"uhat shape: {uhat.shape}, dhat shape: {dhat.shape}, newdist_reshaped shape: {newdist_reshaped.shape}")
                        dhat = np.concatenate([dhat, newdist_reshaped], axis=0)
                    else:
                        uhat = u
                    bestdist = newdist
                    print(f"Updated bestdist in Case B: {bestdist}")
                u[0, k-1] += step[k-1]
                y = (e[k-1, k-1] - u[0, k-1]) / H[k-1, k-1]
                step[k-1] = -step[k-1] - sgn1(step[k-1])
                print(f"Updated u in Case B: {u}, y: {y}, step: {step[k-1]}")
        else:
            if k == n:
                if allnn:
                    if bestdist > 0:
                        dhat /= bestdist
                        dsel = dhat < (1 + epsilon)
                        uhat = selectRows(uhat, dsel[:, 0])
                    else:
                        print(f"Skipping normalization since bestdist is zero.")
                print(f"Final uhat: {uhat}")
                return np.array(uhat)
            else:
                k += 1
                u[0, k-1] += step[k-1]
                y = (e[k-1, k-1] - u[0, k-1]) / H[k-1, k-1]
                step[k-1] = -step[k-1] - sgn1(step[k-1])
                print(f"Updated u in Case C (backtrack): {u}, y: {y}, step: {step[k-1]}")

def closestIndex(R, x=None, allnn=True, epsilon=1e-8):
    G2 = R.T
    U = cholesky(np.dot(G2, G2.T), lower=True)
    flipU = np.diag((np.diag(U) >= 0) * 2 - 1)
    G3 = np.dot(U.T, flipU)
    Q = np.dot(np.dot(flipU, np.linalg.inv(G3)), G2)
    H3 = np.linalg.inv(G3)
    print(H3)
    if x is not None:
        x = x.flatten()  # Ensure x is 1-dimensional
        x3 = np.dot(x, Q.T)
        uhat = closestIndexC(H3, x3, allnn, epsilon)
    else:
        uhat = closestIndexC(H3, allnn=allnn, epsilon=epsilon)
        
    return uhat
