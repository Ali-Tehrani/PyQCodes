import numpy as np
import cvxopt as cvx
import picos as pic
from qutip import rand_dm_ginibre

from PyQCodes._choi import ChoiQutip
from PyQCodes._kraus import DenseKraus


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
    I = np.array([[1., 0.], [0., 1.]], dtype=np.complex128)
    Z = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
    X = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
    Y = np.array([[0., complex(0, -1.)], [complex(0, 1), 0.]], dtype=np.complex128)
    krauss_ops = [I * np.sqrt((1 - p1 - p2 - p3)),
                  X * np.sqrt(p1),
                  Y * np.sqrt(p2),
                  Z * np.sqrt(p3)]
    return krauss_ops


def check_two_sets_of_krauss_are_same(krauss1, krauss2, numb_qubits, dim_in, dim_out, numb=1000):
    r"""Helper function for checking if two kraus operators by checking if they agree."""
    is_same = True
    chann1 = DenseKraus(krauss1, numb_qubits, dim_in, dim_out)
    chann2 = DenseKraus(krauss2, numb_qubits, dim_in, dim_out)
    for _ in range(0, numb):
        # Get random Rho
        rho = np.array(rand_dm_ginibre(2).data.todense())
        rho1 = chann1.channel(rho)
        rho2 = chann2.channel(rho)
        # Compare them
        if np.any(np.abs(rho1 - rho2) > 1e-3):
            is_same = False
            break
    return is_same


def test_creation_of_one_qubit_operators():
    # Depolarizing Channel
    px = 0.1
    lams = np.array([1. - 4. * px, 1. - 4. * px, 1. - 4. * px])

    I = np.eye(2) * np.sqrt(1 - 3. * px)
    X = np.array([[0., 1.], [1., 0.]]) * np.sqrt(px)
    Y = np.array([[0., complex(0, -1.)], [complex(0, 1.), 0.]]) * np.sqrt(px)
    Z = np.array([[1., 0.], [0., -1.]]) * np.sqrt(px)

    # Check with lamda parameters
    channel = ChoiQutip.one_qubit_choi_matrix(lams, np.array([0., 0., 0.]), pauli_errors=False)

    # Check Equivalence to deplorizing channel over random set.
    assert check_two_sets_of_krauss_are_same([I, X, Y, Z], channel.kraus_operators(), [1, 1], 2, 2)

    # Check with Pauli-Errors
    pauli_errors = np.array([px, px, px])
    channel = ChoiQutip.one_qubit_choi_matrix(pauli_errors, np.array([0., 0., 0.]), pauli_errors=True)

    # Check Trace-perserving
    cond = sum([np.conj(x.T).dot(x) for x in channel.kraus_operators()])
    assert np.all(np.abs(cond - np.eye(2)) < 1e-5)

    # Check Equivalence to deplorizing channel over random set.
    assert check_two_sets_of_krauss_are_same([I, X, Y, Z], channel.kraus_operators(), [1, 1], 2, 2)


def test_creation_from_choi_operator():
    r"""Test creation of AnalyticQCode object from choi operator."""
    # Get krauss operators from dephrasure channel
    krauss_ops = set_up_dephrasure_conditions(0.1, 0.2)

    # Construct choi matrix from krauss operators
    choi_matrix = sum([np.outer(np.ravel(x, order="F"),
                                np.conj(np.ravel(x, order="F"))) for x in krauss_ops])
    numb_qubits, dim_in, dim_out = [1, 1], 2, 3
    choi_obj = ChoiQutip(choi_matrix, numb_qubits, dim_in, dim_out)

    # Check if the two constructed krauss operators are the same.
    assert check_two_sets_of_krauss_are_same(krauss_ops, choi_obj.kraus_operators(), numb_qubits,
                                             dim_in, dim_out)


def test_action_of_choi_operator():
    r"""Test that choi operator matches well-known krauss operators."""
    krauss = initialize_pauli_examples(0.1, 0.2, 0.3)
    choi = sum([np.outer(np.ravel(x, "F"),
                         np.conj(np.ravel(x, "F").T)) for x in krauss])
    choi_obj = ChoiQutip(choi, numb_qubits=[1, 1], dim_in=2, dim_out=2)

    for _ in range(0, 1000):
        rho = np.array(rand_dm_ginibre(2).data.todense())
        actual = choi_obj.channel(rho)
        desired = sum([k.dot(rho).dot(np.conj(k).T) for k in krauss])
        assert np.all(np.abs(actual - desired) < 1e-3)

    # Test number of qubits being 2.
    krauss = np.kron(krauss, krauss)
    choi = sum([np.outer(np.ravel(x, "F"),
                         np.conj(np.ravel(x, "F"))) for x in krauss])
    choi_obj = ChoiQutip(choi, numb_qubits=[2, 2], dim_in=2, dim_out=2)

    for _ in range(0, 1000):
        rho = np.array(rand_dm_ginibre(4).data.todense())
        actual = choi_obj.channel(rho)
        desired = sum([k.dot(rho).dot(np.conj(k).T) for k in krauss])
        assert np.all(np.abs(actual - desired) < 1e-3)

    # Test Dephrasure Channe
    krauss = set_up_dephrasure_conditions(0.1, 0.2)
    choi = sum([np.outer(np.ravel(x, "F"),
                         np.conj(np.ravel(x, "F"))) for x in krauss])
    choi_obj = ChoiQutip(choi, [1, 1], 2, 3)

    for _ in range(0, 1000):
        rho = np.array(rand_dm_ginibre(2).data.todense())
        actual = choi_obj.channel(rho)
        desired = sum([k.dot(rho).dot(np.conj(k).T) for k in krauss])
        assert np.all(np.abs(actual - desired) < 1e-3)


def entropyexchange(krauss, rho):
    return np.array([[np.trace(np.conj(k.T).dot(k2).dot(rho)) for k in krauss] for k2 in krauss])


def test_entropy_complementary_channel_of_choi_operator():
    r"""Test the entropy of complementary channel of the choi operator matches entropy exchange."""
    # Dephrasure
    krauss = set_up_dephrasure_conditions(0.1, 0.2)
    choi = sum([np.outer(np.ravel(x, "F"),
                         np.conj(np.ravel(x, "F"))) for x in krauss])
    choi_obj = ChoiQutip(choi, [1, 1], 2, 3)

    for _ in range(0, 1000):
        # Random rho
        rho = rand_dm_ginibre(2).data.todense()
        # Get eigenvalues
        desired = np.linalg.eigvalsh(entropyexchange(krauss, rho))
        actual = np.linalg.eigvalsh(choi_obj.complementary_channel(rho))

        # Assert that eigenvalues are the same.
        i = 0
        for x in actual:
            if np.abs(x) > 1e-8:
                assert np.abs(desired[i] - x) < 1e-5
                i += 1


def test_avg_entanglement_fidelity_ensemble():
    r"""Test the average entanglement fidelity function."""
    # Test on emsemble.
    probs = [1.]
    states = [np.eye(2) / 2.]
    # Test on pauli choi matrix.
    krauss_ops = initialize_pauli_examples(0.1, 0.2, 0.7)
    choi_matrix = sum([np.outer(np.ravel(x, order="F"),
                                np.conj(np.ravel(x, order="F"))) for x in krauss_ops])
    choi_obj = ChoiQutip(choi_matrix, [1, 1], 2, 2)
    actual = choi_obj.average_entanglement_fidelity(probs, states)
    desired = np.ravel(states[0], "F").dot(choi_matrix.dot(np.ravel(states[0], "F")))
    assert np.abs(actual - desired) < 1e-5

    # Test on another ensemble
    probs = [0.25, 0.75]
    states = [np.eye(2), (np.eye(2) + 0.2 * np.array([[0., 1.], [1., 0.]])) / 2.]
    actual = choi_obj.average_entanglement_fidelity(probs, states)
    desired = np.ravel(states[0], "F").dot(choi_matrix.dot(np.ravel(states[0], "F"))) * probs[0]
    desired += np.ravel(states[1], "F").dot(choi_matrix.dot(np.ravel(states[1], "F"))) * probs[1]
    assert np.abs(actual - desired) < 1e-5


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
            X = P.add_variable('X', (2**k * 3**n, 2 **k * 3**n))
            X.value = mat

            # Test partial trace is equivalent to identity.
            args = ChoiQutip.constraint_partial_trace(numb_qubits=[1, 1], dim_in=2, dim_out=3)
            desired = pic.partial_trace(X, *args)
            actual = np.eye(dims[0][0])
            assert np.all(np.abs(desired - actual))
