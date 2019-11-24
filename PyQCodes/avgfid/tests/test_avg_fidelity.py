import numpy as np
import pytest
import picos as pic
import cvxopt as cvx
from PyQCodes.avgfid.avg_fidelity import optimize_decoder_stabilizers
from PyQCodes.chan.channel import ChoiQutip

r"""Test the "avg_fidelity.py file."""


@pytest.mark.slow
def test_optimization_average_fidelity_on_identity_channel():
    r"""Test optimizing decoder on the identity channel."""

    # Five qubit code
    stab_code = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    n = 5
    k = 1
    bin_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                        [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])

    #  Identity channel on a single qubit.
    kraus = [np.eye(2)]
    options = {"maxit": 10}
    result = optimize_decoder_stabilizers(bin_rep, (n, k), kraus, sparse=False, options=options)
    assert np.abs(result["optimal_val"] - 1.) < 1e-2


@pytest.mark.slow
def test_optimization_average_fidelity_on_amplitude_damping():
    r"""Test the optimization of average fidelity on the amplitude damping channel."""
    pass


def set_up_dephrasure_conditions(p, q):
    # Set up the dephrasure channel for the tests bellow.
    krauss_1 = np.array([[1., 0.], [0., 1.], [0., 0.]], dtype=np.complex128)
    krauss_2 = np.array([[1., 0.], [0., -1.], [0., 0.]], dtype=np.complex128)
    krauss_4 = np.array([[0., 0.], [0., 0.], [1., 0.]], dtype=np.complex128)
    krauss_5 = np.array([[0., 0.], [0., 0.], [0., 1.]], dtype=np.complex128)
    krauss_ops = [krauss_1 * np.sqrt((1 - p) * (1 - q)),
                  krauss_2 * np.sqrt((1 - q) * p),
                  np.sqrt(q) * krauss_4,
                  np.sqrt(q) * krauss_5]
    krauss_ops = np.array(krauss_ops, dtype=np.complex128)
    return krauss_ops


def initialize_pauli_examples(p1, p2, p3):
    # Set up pauli channel with errors p1, p2, p3
    identity = np.array([[1., 0.], [0., 1.]], dtype=np.complex128)
    Z = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
    X = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
    Y = np.array([[0., complex(0, -1.)], [complex(0, 1), 0.]], dtype=np.complex128)
    krauss_ops = [identity * np.sqrt((1 - p1 - p2 - p3)),
                  X * np.sqrt(p1),
                  Y * np.sqrt(p2),
                  Z * np.sqrt(p3)]
    return krauss_ops


def test_picos_constraints_for_trace_perserving():
    r"""Test the constraints 'constraint_partial_trace' match picos functionality."""
    # Test over Pauli Chanmel
    for p in np.arange(0., 1., 0.1):
        for p2 in np.arange(0., 1., 0.1):
            kraus = initialize_pauli_examples(p, p2, p)

            # Construct choi matrix from krauss operators
            choi_matrix = sum([np.outer(np.ravel(x, order="F"),
                                        np.conj(np.ravel(x, order="F"))) for x in kraus])

            # Number of input and output qubits, respectively.
            k, n = 1, 1
            numb_rows = 2 ** (n + k)
            numb_cols = 2 ** (n + k)
            assert choi_matrix.size == numb_rows * numb_cols

            mat = cvx.matrix(choi_matrix)
            dims = [(2 ** k, 2 ** k), (2 ** n, 2 ** n)]
            P = pic.Problem()
            X = P.add_variable('X', (4, 4))
            X.value = mat

            # Test partial trace is equivalent to identity.
            args = ChoiQutip.constraint_partial_trace(numb_qubits=[1, 1], dim_in=2, dim_out=2)
            desired = pic.partial_trace(X, *args)
            actual = np.eye(dims[0][0])
            assert np.all(np.abs(desired - actual))

    # Test over dephrasure channel.
    for p in np.arange(0, 1, 0.1):
        for p2 in np.arange(0, 1, 0.1):
            kraus = set_up_dephrasure_conditions(p, p2)

            # Construct choi matrix from krauss operators
            choi_matrix = sum([np.outer(np.ravel(x, order="F"),
                                        np.conj(np.ravel(x, order="F"))) for x in kraus])

            # Number of input and output qubits, respectively.
            k, n = 1, 1
            numb_rows = 2 ** (n + k)
            numb_cols = 3 ** (n + k)
            assert choi_matrix.size == numb_rows * numb_cols

            mat = cvx.matrix(choi_matrix)
            dims = [(2 ** k, 2 ** k), (3 ** n, 3 ** n)]
            P = pic.Problem()
            X = P.add_variable('X', (2**k * 3**n, 2**k * 3**n))
            X.value = mat

            # Test partial trace is equivalent to identity.
            args = ChoiQutip.constraint_partial_trace(numb_qubits=[1, 1], dim_in=2, dim_out=3)
            desired = pic.partial_trace(X, *args)
            actual = np.eye(dims[0][0])
            assert np.all(np.abs(desired - actual))

    # Test over higher dimensions.
    kraus = np.array(set_up_dephrasure_conditions(0.1, 0.2))
    kraus = np.kron(kraus, kraus)

    # Construct choi matrix from krauss operators
    choi_matrix = sum([np.outer(np.ravel(x, order="F"),
                                np.conj(np.ravel(x, order="F"))) for x in kraus])

    # Number of input and output qubits, respectively.
    k, n = 2, 2
    numb_rows = 2 ** (n + k)
    numb_cols = 3 ** (n + k)
    assert choi_matrix.size == numb_rows * numb_cols

    mat = cvx.matrix(choi_matrix)
    dims = [(2 ** k, 2 ** k), (3 ** n, 3 ** n)]
    P = pic.Problem()
    X = P.add_variable('X', (2 ** k * 3 ** n, 2 ** k * 3 ** n))
    X.value = mat

    # Test partial trace is equivalent to identity.
    args = ChoiQutip.constraint_partial_trace(numb_qubits=[2, 2], dim_in=2, dim_out=3)
    desired = pic.partial_trace(X, *args)
    actual = np.eye(dims[0][0])
    assert np.all(np.abs(desired - actual))


if __name__ == "__main__":
    test_picos_constraints_for_trace_perserving()
    test_optimization_average_fidelity_on_identity_channel()
