import numpy as np
from PyQCodes.chan._choi import ChoiQutip
from scipy.sparse import coo_matrix, issparse, eye, kron, isspmatrix_coo

# Semi-definite Solver
import picos as pic
import cvxopt as cvx


r""" 
Contains optimization routines for optimizing over density matrices 
and over choi-matrices, as semi-definite problems.
"""


__all__ = ["positive_coherent_info_callback"]


def positive_coherent_info_callback(x_k, func_val, convergence):
    print("func val ", func_val)
    if func_val < -1e-8:
        return True
    return False


def _lipschitz_optimizing_condition():
    pass


def initial_guesses_lipschitz(f, nsize, rank, param, chan, n, numb_samples=10, numb_tries=100):
    r"""


    Parameters
    ----------
    f : callable
        Objective, Lipschitz Function that takes input from argument 'rho_sampler'.

    param : callable
        Function that returns uniformly random density states.

    numb_samples : int
        Number of samples returned.

    numb_tries : int
        Number of random samples to try.

    Returns
    -------


    References
    ----------
    Based on the 2017 paper, "Global Optimization of Lipschitz Functions"
    """
    # Obtain single random vectors turn into density matrix and evaluate on objective function.
    samples_vec = [param.random_vectors(nsize, rank)]
    print(nsize, rank)
    samples_rho = [param.rho_from_vec(samples_vec[0], nsize, rank)]
    func_evals = [-f(samples_vec[0], nsize, rank, param, chan, n)]
    counter = 0
    success = False
    f_max = func_evals[0]
    while (counter < numb_tries) and len(samples_vec) < numb_samples:
        vec = param.random_vectors(nsize, rank)
        rho = param.rho_from_vec(vec, nsize, rank)

        bounds = [func_evals[i] + np.sum(np.abs(rho - samples_rho[i]))
                  for i in range(0, len(samples_vec))]
        if f_max <= np.min(bounds):
            samples_vec.append(vec)
            samples_rho.append(rho)
            func_evals.append(-f(vec, nsize, rank, param, chan, n))
            f_max = np.max(func_evals)
        counter += 1

    if len(samples_vec) == numb_samples:
        success = True
    return samples_vec, func_evals, success
