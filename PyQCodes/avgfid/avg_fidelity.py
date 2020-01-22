r"""
######################
### IMPORTANT ########
######################

This file has the GNU license due to using the python package PICOS.
Please be careful.

PyQCodes - Software for investigating coherent information and
optimization-based quantum error-correcting codes.
PyQCodes is jointly owned equally by the University of Guelph (and its employees)
and Huawei, funded through the Huawei Innovation Research.
"""
import numpy as np
from PyQCodes.chan._choi import ChoiQutip
from PyQCodes.chan.channel import AnalyticQChan
from PyQCodes.codes import StabilizerCode
from scipy.sparse import coo_matrix, issparse, isspmatrix_coo

# Semi-definite Solver
import picos as pic
import cvxopt as cvx

__all__ = ["optimize_decoder", "optimize_decoder_stabilizers"]


def optimize_decoder(encoder, kraus_chan, numb_qubits, dim_in, dim_out, objective="coherent",
                     sparse=False, param_dens="over", param_decoder="", options=None):
    #TODO:

    # Set up stabilizer code, pauli-channel error and update the channel to match encoder.
    error_chan = AnalyticQChan(kraus_chan, numb_qubits, dim_in, dim_out)
    error_chan.krauss.update_kraus_operators(code_param[0] // code_param[1])

    # Kraus operators for channel composed with encoder.
    kraus_chan = [x.dot(encoder) for x in error_chan.nth_kraus_operators]

    numb_qubits = [code_param[1], code_param[0]]
    dim_in, dim_out = 2, 2
    result = _optimize_average_entang_fid(kraus_chan, numb_qubits, dim_in, dim_out, options=options)
    return result


def optimize_decoder_stabilizers(stabilizer, code_param, kraus_chan, sparse=False, options=None):
    r"""
    Optimizes the average entanglement fidelity with respect to space of all decoders.

    Parameters
    ----------
    stabilizer_group : np.ndarray or list
            Binary Representation of stabilizer groups or list of pauli strings of the encoder.
    code_param : (int, int)
        The parameters (n, k) of the code of the stabilizer. The stabilizer code maps k qubits to
        n qubits.
    kraus_chan : list or np.ndarray
        The kraus operators of the channel acting on k-qubits.
    sparse : bool
        Whether to model the kraus operators and encoder as sparse matrices.
    options : dict
        Dictionary with the following keys, 'maxit', 'feastol', 'abstol' and 'reltol'. Default
        option for each respectively are 50, 1e-3, 1e-3, 1e-3.  These options are found further in
        'picos.problem.set_options'.

    Returns
    -------
    dict :
        The result is a dictionary with fields:

            optimal_recov : picos.Variable
                The choi matrix of the recovery operation in picos.Variable. Printing it gives
                the final result.
            optimal_val : float
                The optimal value of average fidelity.
            time : float
                The time in seconds that it took.
            status : bool
                Status of convergence of the optimizer.

    Notes
    -----
    - Assumes that all dimensions of single-particle hilbert space (input and output of channel)
        is 2.

    """
    # Set up stabilizer code, pauli-channel error and update the channel to match encoder.
    stab = StabilizerCode(stabilizer, code_param[0], code_param[1])
    error_chan = AnalyticQChan(kraus_chan, [1, 1], 2, 2, sparse=sparse)
    error_chan.krauss.update_kraus_operators(code_param[0] // code_param[1])

    # Get Kraus Operator for encoder.
    encoder = stab.encode_krauss_operators(sparse=sparse)

    # Kraus operators for channel composed with encoder.
    if sparse:
        kraus_chan = [x.tocsr().dot(encoder) for x in error_chan.nth_kraus_operators]
    else:
        kraus_chan = [x.dot(encoder) for x in error_chan.nth_kraus_operators]

    numb_qubits = [code_param[1], code_param[0]]
    dim_in, dim_out = 2, 2
    result = _optimize_average_entang_fid(kraus_chan, numb_qubits, dim_in, dim_out, options=options)
    return result


def _convert_sparse_matrix_into_cvx_opt(mat):
    mat = mat.tocoo()
    return cvx.spmatrix(mat.data, mat.row, mat.col, size=mat.shape)


def _optimize_average_entang_fid(krauss, numb_qubits, dim_in, dim_out, encoder=False, verbose=True,
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
        option for each respectively are 50, 1e-3, 1e-3, 1e-3.  These options are found further in
        'picos.problem.set_options'.

    Returns
    -------
    dict :
        The result is a dictionary with fields:

            optimal_recov : picos.Variable
                The choi matrix of the encoder/recovery operation in picos.Variable.
                Try printing it to see the result.
            optimal_val : float
                The optimal value of average fidelity.
            time : float
                The time in seconds that it took.
            status : bool
                Status of convergence of the optimizer.

    Notes
    -----
    -- Kletcher in his book uses row-stacking, whereas Qutip and this software uses
    column-stacking.

    References
    ----------
    -- See "Andrew Kletcher's Chapter Two thesis, "Channel-Adapted Quantum Error Correction"."
    -- Alternately, See Lidar's 'Quantum Error-Correction' book Chapter 13.

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
        print("The size is ", krauss[0].size)
        print(krauss[0].shape)
        print(type(krauss[0]))
        C_e_sparse = sum([np.outer(k.conj().reshape((np.prod(k.shape), 1), order="F"),
                                   k.conj().reshape((np.prod(k.shape), 1), order="F"))
                          for k in krauss]) / input_dim**2
    else:
        C_e = sum([np.outer(np.ravel(np.conj(k).T, order="F"),
                            np.ravel(np.conj(k).T, order="F")) for k in krauss]) / input_dim**2
        C_e_sparse = coo_matrix(C_e)  # Turn to sparse array

    # Set up objective function as average fidelity wrt to maximally mixed state ensemble.
    sparse_mat = _convert_sparse_matrix_into_cvx_opt(C_e_sparse)
    P.set_objective('max', (X | sparse_mat))

    # Solve
    if options is None:
        options = dict({})
    P.set_option('tol', 1e-4)
    P.set_option('maxit', 50)  # Default is 100
    P.set_option('feastol', 1e-3)  # Default s 1e-7
    P.set_option('abstol', 1e-3)  # Default is 1e-7
    P.set_option('reltol', 1e-3)  # Default is 1e-6
    if "maxit" in options:
        P.set_option("maxit", options["maxit"])
    if "feastol" in options:
        P.set_option("feastol", options["feastol"])
    if "abstol" in options:
        P.set_option("abstol", options["abstol"])
    if "reltol" in options:
        P.set_option("reltol", options["reltol"])
    solution = P.solve(verbose=verbose, solver='cvxopt')
    result =  {"status": solution["status"], "time" : solution["time"], "optimal_recov": X,
               "optimal_val" : P.obj_value()}
    return result
