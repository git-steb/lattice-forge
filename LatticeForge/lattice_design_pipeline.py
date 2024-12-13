import numpy as np
import itertools
import scipy.linalg as la
from sympy import I, sqrt, eye

import logging

# Configure logging at the beginning of your script
logging.basicConfig(
    level=logging.WARN,  # Set to DEBUG to capture all levels of log messages
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs will be output to the console
    ]
)
#logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# Helper function for generating filenames
def genfilename(N, detK, suffix):
    return f"lattices_dim{N}_det{detK}_{suffix}.npy"

def verifyK(K, detK=2, eps=1e-2):
    """
    Verify if the subsampling matrix K permits an equivalent subsampling lattice of determinant detK.

    Parameters:
    - K (np.ndarray): Subsampling matrix (must be an integer matrix).
    - detK (float): Desired determinant value.
    - eps (float): Tolerance for verification.

    Returns:
    - bool: True if K passes the verification, False otherwise.
    """
    logging.debug(f"Starting verifyK with K:\n{K}, detK={detK}, eps={eps}")
    ND = len(K)
    logging.debug(f"Matrix dimension (ND): {ND}")

    try:
        if ND % 2 == 1:
            logging.debug("Matrix dimension is odd.")
            e = np.linalg.eigvals(K)
            logging.debug(f"Eigenvalues of K: {e}")
            eigen_det_diff = np.abs(np.abs(e ** ND) - detK)
            logging.debug(f"Difference between |e^ND| and detK: {eigen_det_diff}")
            if np.max(eigen_det_diff) > eps:
                logging.debug("Maximum eigenvalue determinant difference exceeds epsilon.")
                return False
        else:
            logging.debug("Matrix dimension is even.")
            c = np.poly(K)
            logging.debug(f"Characteristic polynomial coefficients: {c}")
            zeroind = list(range(1, ND // 2)) + list(range(ND // 2 + 1, ND))
            logging.debug(f"Indices expected to be zero in polynomial: {zeroind}")
            if zeroind:
                zero_values = c[zeroind]
                logging.debug(f"Polynomial coefficients at zero indices: {zero_values}")
                if np.max(np.abs(zero_values)) > eps:
                    logging.debug("Non-zero coefficients found at zero indices exceeding epsilon.")
                    return False
            Cf = c[ND // 2]
            logging.debug(f"Middle coefficient Cf: {Cf}")
            C = round(Cf)
            logging.debug(f"Rounded Cf (C): {C}")
            if abs(C - Cf) > eps:
                logging.debug("Difference between C and Cf exceeds epsilon.")
                return False
            D = c[-1]
            logging.debug(f"Last coefficient (D): {D}")
            computed_detK = (-1)**ND * D
            logging.debug(f"Computed determinant (computed_detK): {computed_detK}")
            if abs(computed_detK - detK) > eps:
                logging.debug("Difference between computed_detK and detK exceeds epsilon.")
                return False
    except Exception as e:
        logging.error(f"An error occurred in verifyK: {e}")
        return False

    logging.debug("Matrix K passed verification.")
    return True

def formJ(N, symbolic=False):
    logging.debug(f"Creating congruency transform matrix J with N={N}, symbolic={symbolic}")
    
    if symbolic:
        sqrt2 = sqrt(2)
        J2 = np.array([[1, I], [1, -I]]) / sqrt2
        J = eye(N)
        logging.debug(f"Symbolic J2:\n{J2}")
    else:
        sqrt2 = np.sqrt(2)
        J2 = np.array([[1, 1j], [1, -1j]]) / sqrt2
        J = np.eye(N, dtype=complex)
        logging.debug(f"Numeric J2:\n{J2}")

    for l in range(N - 2, -1, -2):
        logging.debug(f"Updating J[{l}:{l + 2}, {l}:{l + 2}] with J2")
        J[l:l + 2, l:l + 2] = J2

    logging.debug(f"Final J matrix:\n{J}")
    return J

def formlattice(K, J=None, invJ=None):
    logging.debug(f"Forming lattice with K:\n{K}")
    N = len(K)
    if J is None:
        logging.debug("No J provided. Forming J using formJ.")
        J = formJ(N)
    if invJ is None:
        logging.debug("No invJ provided. Computing inverse of J.")
        try:
            invJ = np.linalg.inv(J)
        except np.linalg.LinAlgError as e:
            logging.error(f"Failed to invert J: {e}")
            raise

    logging.debug(f"J:\n{J}")
    logging.debug(f"invJ:\n{invJ}")

    try:
        D, V = np.linalg.eig(K)
        logging.debug(f"Eigenvalues (D): {D}")
        logging.debug(f"Eigenvectors (V):\n{V}")

        # Ensure P and invP are invertible
        P = np.eye(N, dtype=complex)
        invP = np.linalg.inv(P)
        logging.debug(f"P matrix:\n{P}")
        logging.debug(f"invP matrix:\n{invP}")

        V = V @ invP
        logging.debug(f"V after P transformation:\n{V}")
        D_matrix = P @ np.diag(D) @ invP
        logging.debug(f"D as diagonal matrix:\n{D_matrix}")

        R = np.linalg.inv(V @ J)
        logging.debug(f"Lattice basis R:\n{R}")
        Q = invJ @ D_matrix @ J
        logging.debug(f"Similarity transform Q:\n{Q}")

    except np.linalg.LinAlgError as e:
        logging.error(f"Linear algebra error in formlattice: {e}")
        raise

    return R, Q

def searchK(N, detK=2, value_range=(-5, 5), nsamp=1000, save=True):
    """
    Search for subsampling matrices K and verify them.

    Parameters:
    - N (int): Lattice dimension.
    - detK (int): Determinant of permissible K.
    - value_range (tuple): Range of values for the entries of K.
    - nsamp (int): Number of random samples for non-exhaustive search.
    - save (bool): Whether to save the results.

    Returns:
    - list: List of valid subsampling matrices.
    """
    results = []
    min_val, max_val = value_range
    total_possible = (max_val - min_val + 1) ** (N * N)
    logging.debug(f"Starting searchK with N={N}, detK={detK}, value_range={value_range}, nsamp={nsamp}, save={save}")
    logging.debug(f"Total possible matrices in range: {total_possible}")

    for i in range(1, nsamp + 1):
        K = np.random.randint(min_val, max_val + 1, size=(N, N))
        if i % 100 == 0 or i == 1 or i == nsamp:
            logging.debug(f"Sample {i}/{nsamp}: Matrix K=\n{K}")
        if verifyK(K, detK):
            logging.debug(f"Matrix K at sample {i} passed verification.")
            results.append(K)
        else:
            logging.debug(f"Matrix K at sample {i} failed verification.")

    if save:
        filename = genfilename(N, detK, "K")
        np.save(filename, results)
        logging.debug(f"Saved {len(results)} subsampling matrices to {filename}")

    logging.debug(f"searchK completed with {len(results)} valid matrices found.")
    return results

def doall(N=2, detK=2, forcerecompute=False, usecompangen=True, ngen=1, optimize=True):
    logging.debug(f"Running doall with N={N}, detK={detK}, forcerecompute={forcerecompute}, usecompangen={usecompangen}, ngen={ngen}, optimize={optimize}")
    filename = genfilename(N, detK, "KRQE")
    
    if not forcerecompute:
        try:
            results = np.load(filename, allow_pickle=True)
            logging.debug(f"Loaded precomputed results from {filename}")
        except FileNotFoundError:
            logging.warning(f"File {filename} not found. Forcing recompute.")
            forcerecompute = True
    
    if forcerecompute:
        if usecompangen:
            logging.debug("Using companion similarity to generate K")
            results = searchK(N, detK)
        elif N <= 4:
            results = searchK(N, detK)
        else:
            logging.debug("Using large sample search for high dimensions.")
            results = searchK(N, detK, nsamp=10**6)
        
        np.save(filename, results)
        logging.debug(f"Saved results to {filename}")
    
    for idx, K in enumerate(results, start=1):
        logging.debug(f"Processing matrix {idx}/{len(results)}:\n{K}")
        try:
            R, Q = formlattice(K)
            logging.debug(f"Lattice Basis R:\n{R}\n")
        except np.linalg.LinAlgError:
            logging.error(f"Linear algebra error with matrix K:\n{K}")
            continue

    logging.debug("doall completed successfully.")
    return results

def enumerate_matrices(N, detK, value_range):
    """
    Enumerate all possible N x N integer matrices within a given range and verify their determinants.

    Parameters:
    - N (int): Dimension of the matrix.
    - detK (int or float): Desired determinant value.
    - value_range (tuple): Range for matrix entries (min, max).

    Returns:
    - list: List of valid subsampling matrices.
    """
    min_val, max_val = value_range
    results = []
    total = (max_val - min_val + 1) ** (N * N)
    logging.debug(f"Starting enumeration with N={N}, detK={detK}, value_range={value_range}")
    logging.debug(f"Total possible matrices: {total}")

    # Generate all possible combinations of entries
    for idx, entries in enumerate(itertools.product(range(min_val, max_val + 1), repeat=N*N), start=1):
        if idx % 100000 == 0 or idx == 1 or idx == total:
            logging.debug(f"Enumerating matrix {idx}/{total}")
        K = np.array(entries).reshape(N, N)
        if verifyK(K, detK):
            logging.debug(f"Matrix {idx} passed verification.")
            results.append(K)
        else:
            logging.debug(f"Matrix {idx} failed verification.")

    logging.debug(f"Enumeration completed with {len(results)} valid matrices found.")
    return results


# Example usage
if __name__ == "__main__":
    doall(N=3, detK=2)
