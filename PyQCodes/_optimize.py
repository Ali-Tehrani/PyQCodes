import numpy as np
from PyQCodes._choi import ChoiQutip
from scipy.sparse import coo_matrix, issparse, eye, kron, isspmatrix_coo

# Semi-definite Solver
import picos as pic
import cvxopt as cvx
# General Solver
from scipy.optimize import minimize

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


def solve_for_matrix_coefficients(matrix):
    r"""
    Decomposes a matrix based on single pauli operators.

    Only works for 2x2 matrices.
    """
    assert matrix.shape == (2, 2)
    coeff_matrix = np.array([[0., 0., 1.],
                             [1., complex(0, -1.), 0],
                             [1., complex(0, 1), 0],
                             [0., 0., -1.]])
    vect_mat = np.ravel(matrix)
    coeffs = np.linalg.solve(coeff_matrix, vect_mat)
    return coeffs


def _convert_sparse_matrix_into_cvx_opt(mat):
    mat = mat.tocoo()
    return cvx.spmatrix(mat.data, mat.row, mat.col, size=mat.shape)


def optimize_average_entang_fid(krauss, numb_qubits, dim_in, dim_out, encoder=False, verbose=True,
                                options=None):
    r"""
    Optimizes the average entanglement fidelity of a quantum channel wrt decoder/encoder.

    Uses the choi-matrix formulation and assumes that the emsemble is the maximally
    mixed state with probability one. This is primarly used for channel-adapted
    quantum error correcting codes. See Notes.

    Parameters
    ---------
    kraus : list or np.array
        List of arrays each representing the kraus operators of either the .math.'\Phi \circ E',
        where E is the encoding operator or .math.'R \circ \Phi', where R is the recovery operator.

    numb_qubits : (int, int)
        Tuple of two integers M, N, where M represents teh number of qubits the recovery operator
        takes and N represents the number of qubits the recovery operators "sends out". If
        'encoder' is true then this represents the encoder operator.

    dim_in : int
        Dimension of single-particle, input hilbert space of the recovery operator. For qubit
        particles, dimension of hilbert space is two.  If 'encoder' is true then this represents
        the encoder operator.

    dim_out : int
        Dimension of single-particle, output hilbert space of the recovery operator. For qubit
        particles, dimension of hilbert space is two.  If 'encoder' is true then this represents
        the encoder operator.

    encoder : bool
        True, if optimization is done over the space of all encoding operator, where the
        recovery operator is fixed. Default is false.

    verbose : bool
        If true, then the solver is told to be verbose. Default is True.

    options : dict
        Dictionary with the following keys, 'maxit', 'feastol', 'abstol' and 'reltol'. Default
        option for each respectively are 50, 1e-5, 1e-5, 1e-5.  These options are found further in
        'picos.problem.set_options'.

    Returns
    -------


    Notes
    -----
    -- Kletcher in his book uses row-stacking, whereas Qutip and this software uses
    column-stacking.

    References
    ----------
    -- See "Andrew Kletcher's Chapter Two thesis, "Channel-Adapted Quantum Error Correction"."
    -- Alternately, See Lidar's 'Quantum Error-Correction' book.

    Examples
    --------
    # Example taken from Reference.
    # Construct Amplitude Damping Channel
    damp = 0.05
    k0 = np.array([[1., 0.], [0., np.sqrt(1 - damp)]])
    k1 = np.array([[0., np.sqrt(damp)], [0., 0.]])
    kraus = [k0, k1]

    # Get encoder of kraus operators to n-qubits.
    encoder = np.array()

    # Construct Ampltitude Plus Encoder
    # Update kraus operators to correspond to channel tensored five times.
    kraus = [kraus operators composed with encoder]
    optimize_average_entang_fid(kraus, (5, 1), 2, 2)
    """
    P = pic.Problem()
    input_dim = dim_in ** numb_qubits[0]  # Dimension of hilbert space of input.
    numb_rows_cols = input_dim * dim_out ** numb_qubits[1]  # Number of rows/cols of choi-matrix.

    # Optimize with respect to encoder or decoder.
    X = P.add_variable('X', (numb_rows_cols, numb_rows_cols), 'hermitian')
    # Obtain the constraints needed to partial trace the right dimensions.
    if encoder:
        args = ChoiQutip.constraint_partial_trace(numb_qubits, dim_in, dim_out)
    else:
        args = ChoiQutip.constraint_partial_trace(numb_qubits, dim_in, dim_out)

    # Add constraints that choi matrix is positive-definite and trace-perserving, respecively.
    P.add_constraint(X >> 0)
    P.add_constraint(pic.partial_trace(X, *args) == 'I')

    # Get choi-matrix of the kraus operators on the identity matrix using column-vectorization.
    print(type(krauss), issparse(krauss[0]), type(krauss[0]), isspmatrix_coo(krauss[0]))
    if issparse(krauss[0]):  # If kraus operators are already sparse.
        C_e_sparse = sum([np.outer(k.conj().reshape((k.size, 1), order="F"),
                                   k.conj().reshape((k.size, 1), order="F"))
                          for k in krauss]) / input_dim**2
    else:
        C_e = sum([np.outer(np.ravel(np.conj(k).T, order="F"),
                            np.ravel(np.conj(k).T, order="F")) for k in krauss]) / input_dim**2
        C_e_sparse = coo_matrix(C_e)  # Turn to sparse array

    # Set up objective function as average fidelity wrt to maximally mixed state ensemble.
    sparse_mat = _convert_sparse_matrix_into_cvx_opt(C_e_sparse)
    P.set_objective('max', (X | sparse_mat))

    # Solve
    P.set_option('tol', 1e-5)
    P.set_option('maxit', 50)  # Default is 100
    P.set_option('feastol', 1e-5)  # Default s 1e-7
    P.set_option('abstol', 1e-5)  # Default is 1e-7
    P.set_option('reltol', 1e-5)  # Default is 1e-6
    P.solve(verbose=verbose, solver='cvxopt')
    return X
