from PyQCodes._kraus import DenseKraus, SparseKraus
from qutip import rand_dm_ginibre
from sparse import SparseArray

import numpy as np
from scipy.linalg import block_diag
from numpy.testing import assert_array_almost_equal, assert_raises

r"""Tests for 'QCorr.SparseKraus'."""


#########################
# Test Helper Functions #
#########################

def initialize_dephrasure_examples(p, q):
    # Set up the dephrasure channel for the tests bellow.
    krauss_1 = np.array([[1., 0.], [0., 1.], [0., 0.]], dtype=np.complex128)
    krauss_2 = np.array([[1., 0.], [0., -1.], [0., 0.]], dtype=np.complex128)
    krauss_4 = np.array([[0., 0.], [0., 0.], [1., 0.]], dtype=np.complex128)
    krauss_5 = np.array([[0., 0.], [0., 0.], [0., 1.]], dtype=np.complex128)
    krauss_ops = [krauss_1 * np.sqrt((1 - p) * (1 - q)),
                  krauss_2 * np.sqrt((1 - q) * p),
                  np.sqrt(q) * krauss_4,
                  np.sqrt(q) * krauss_5]
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


def init_of_krauss_and_exchange_array(k_ops):
    desired_krauss = np.array([k_ops[0], k_ops[1], k_ops[2], k_ops[3]])
    desired_exchange = np.array([[np.conj(x.T).dot(y) for y in k_ops] for x in k_ops])
    return desired_krauss, desired_exchange


######################
### Test Functions ###
######################

def test_initialization_of_krauss_operators_sparse():
    r"""Test sparse krauss operators and exchange array have the right shape."""
    # Test it on dephrasure channel.
    p, q = 0.25, 0.1
    k_ops = initialize_dephrasure_examples(p, q)
    numb_krauss = len(k_ops)
    desired_krauss, desired_exchange = init_of_krauss_and_exchange_array(k_ops)

    krauss = SparseKraus(k_ops, [1, 1], 2, 3)

    # Test Krauss operators are in the right format.
    actual = krauss.nth_kraus_ops
    assert issubclass(type(actual), SparseArray)
    actual = actual.todense()
    assert actual.shape == (numb_krauss, 3, 2)
    assert_array_almost_equal(actual, desired_krauss)

    # Test Array for entropy exchange is in the right format.
    actual = krauss.nth_kraus_exch
    assert issubclass(type(actual), SparseArray)
    actual = actual.todense()
    assert actual.shape == (numb_krauss, numb_krauss, 2, 2)
    assert_array_almost_equal(actual, desired_exchange)

    # Test wrong dimensions
    assert_raises(AssertionError, SparseKraus, k_ops, [1, 1], 2, 2)
    assert_raises(AssertionError, SparseKraus, k_ops, [1, 2], 2, 3)
    assert_raises(AssertionError, SparseKraus, k_ops, [1, 1], 3, 2)
    k_ops = [np.array([[1., 0.], [0., 1.]])]
    assert_raises(AssertionError, SparseKraus, k_ops, [1, 1], 2, 3)


def test_initialization_of_krauss_operators_dense():
    r"""Test dense krauss operators and exchange array have the right shape."""
    p, q = 0.25, 0.1
    k_ops = initialize_dephrasure_examples(p, q)
    numb_krauss = len(k_ops)
    desired_krauss, desired_exchange = init_of_krauss_and_exchange_array(k_ops)

    krauss = DenseKraus(k_ops, [1, 1], 2, 3)

    # Test Krauss operators are in the right format.
    actual = krauss.nth_kraus_ops
    assert actual.shape == (numb_krauss, 3, 2)
    assert_array_almost_equal(actual, desired_krauss)

    # Test Array for entropy exchange is in the right format.
    actual = krauss.nth_kraus_exch
    assert actual.shape == (numb_krauss, numb_krauss, 2, 2)
    assert_array_almost_equal(actual, desired_exchange)

    # Test wrong dimensions being provided gives error.
    assert_raises(AssertionError, DenseKraus, k_ops, [1, 1], 2, 2)
    assert_raises(AssertionError, DenseKraus, k_ops, [1, 2], 2, 3)
    assert_raises(AssertionError, DenseKraus, k_ops, [1, 1], 3, 2)
    k_ops = [np.array([[1., 0.], [0., 1.]])]
    assert_raises(AssertionError, DenseKraus, k_ops, [1, 1], 2, 3)


def test_update_krauss_operator_function_dense():
    r"""Test the function 'DenseKraus.update_nth_krauss_ops'."""
    p, q = 0.2, 0.4

    single_krauss_ops = initialize_dephrasure_examples(p, q)
    krauss_ops = single_krauss_ops.copy()

    krauss = DenseKraus(single_krauss_ops, [1, 1], 2, 3)

    # Test increasing to n = 4
    for i in range(1, 4):
        krauss_ops = np.kron(single_krauss_ops, krauss_ops)
    krauss.update_kraus_operators(4)
    assert_array_almost_equal(krauss.nth_kraus_ops, krauss_ops)

    # Test decreasing to n = 3
    krauss_ops = np.kron(single_krauss_ops, single_krauss_ops)
    krauss.update_kraus_operators(2)
    assert_array_almost_equal(krauss.nth_kraus_ops, krauss_ops)


def test_update_krauss_operator_function_sparse():
    r"""Test the function 'SparseKraus._update_nth_krauss_ops'."""
    # Test on dephrasure channel.
    p, q = 0.2, 0.4
    single_krauss_ops = initialize_dephrasure_examples(p, q)
    krauss = SparseKraus(single_krauss_ops, [1, 1], 2, 3)

    # Test increasing to n = 4
    desired = single_krauss_ops.copy()
    for i in range(1, 4):
        desired = np.kron(single_krauss_ops, desired)
    krauss.update_kraus_operators(4)
    assert_array_almost_equal(krauss.nth_kraus_ops.todense(), desired)

    # Test decreasing to n = 3
    krauss_ops = np.kron(single_krauss_ops, single_krauss_ops)
    krauss.update_kraus_operators(2)
    assert_array_almost_equal(krauss.nth_kraus_ops.todense(), krauss_ops)


def test_channel_method_using_dephrasure_sparse():
    r"""Test the channel from 'krauss.SparseKraus.channel' using dephrasure channel."""
    p, q = 0.2, 0.4
    k_ops = initialize_dephrasure_examples(p, q)
    krauss_ops = np.array(k_ops).copy()
    spkrauss = SparseKraus(k_ops, [1, 1], 2, 3)

    for n in range(1, 4):
        if n != 1:
            krauss_ops = np.kron(np.array(k_ops), krauss_ops)

        # Get random density state.
        rho = np.array(rand_dm_ginibre(2**n).data.todense())
        assert np.all(krauss_ops.shape[2] == rho.shape[0])

        # Multiply each krauss operators individually
        desired = np.zeros((3**n, 3**n), dtype=np.complex128)
        for krauss in krauss_ops:
            temp = krauss.dot(rho).dot(krauss.T.conj())
            desired += temp

        spkrauss.update_kraus_operators(n)
        actual = spkrauss.channel(rho)
        assert_array_almost_equal(actual, desired)


def test_channel_method_using_dephrasure_dense():
    r"""Test the channel from 'krauss.DenseKraus.channel' using dephrasure example."""
    p, q = 0.2, 0.4
    k_ops = initialize_dephrasure_examples(p, q)
    krauss_ops = np.array(k_ops).copy()
    dkrauss = DenseKraus(k_ops, [1, 1], 2, 3)

    for n in range(1, 4):
        if n != 1:
            krauss_ops = np.kron(np.array(k_ops), krauss_ops)

        # Get random density state.
        rho = np.array(rand_dm_ginibre(2**n).data.todense())
        assert np.all(krauss_ops.shape[2] == rho.shape[0])

        # Multiply each krauss operators individually
        desired = np.zeros((3**n, 3**n), dtype=np.complex128)
        for krauss in krauss_ops:
            temp = krauss.dot(rho).dot(krauss.T.conj())
            desired += temp

        dkrauss.update_kraus_operators(n)
        actual = dkrauss.channel(rho)
        assert_array_almost_equal(actual, desired)


def entropy_exchange_helper_assertion(krauss, single_krauss_ops):
    # Helps check that entropy exchange matrix entries matches definition.
    krauss_ops = np.array(single_krauss_ops).copy()

    for n in range(1, 4):
        krauss.update_kraus_operators(n)
        if n != 1:
            krauss_ops = np.kron(single_krauss_ops, krauss_ops)

        # Get random density state.
        rho = np.array(rand_dm_ginibre(2 ** n).data.todense())
        assert np.all(krauss_ops.shape[2] == rho.shape[0])

        actual = krauss.entropy_exchange(rho, n)[0]
        assert actual.shape == (4**n, 4**n)
        for i in range(0, 4**n):
            for j in range(0, 4**n):
                desired = np.trace(krauss_ops[i].dot(rho.dot(np.conjugate(krauss_ops[j]).T)))
                assert np.abs(actual[i, j] - desired) < 1e-10


def test_entropy_exchange_dephrasure_channel_dense():
    p, q = 0.2, 0.4
    single_krauss_ops = initialize_dephrasure_examples(p, q)
    dkrauss = DenseKraus(single_krauss_ops, [1, 1], 2, 3)
    entropy_exchange_helper_assertion(dkrauss, single_krauss_ops)


def test_entropy_exchange_dephrasure_channel_sparse():
    p, q = 0.2, 0.4
    single_krauss_ops = initialize_dephrasure_examples(p, q)
    spskrauss = SparseKraus(single_krauss_ops, [1, 1], 2, 3)
    entropy_exchange_helper_assertion(spskrauss, single_krauss_ops)


def test_sparse_krauss_versus_dense_krauss():
    r"""Test the sparse krauss operators versus dense krauss operators."""
    p, q = 0.2, 0.4

    k_ops = initialize_dephrasure_examples(p, q)
    spkrauss = SparseKraus(k_ops, [1, 1], 2, 3)
    dKrauss = DenseKraus(k_ops, [1, 1], 2, 3)

    for n in range(1, 4):
        # Update channel to correpond to n.
        spkrauss.update_kraus_operators(n)
        dKrauss.update_kraus_operators(n)

        # Test nth krauss operators
        assert issubclass(type(spkrauss.nth_kraus_ops), SparseArray)
        spkrauss_nth_krauss = spkrauss.nth_kraus_ops.todense()
        dkrauss_nth_krauss = dKrauss.nth_kraus_ops
        assert_array_almost_equal(spkrauss_nth_krauss, dkrauss_nth_krauss)

        # Test exchange array
        assert issubclass(type(spkrauss.nth_kraus_exch), SparseArray)
        spkrauss_exchange = spkrauss.nth_kraus_exch.todense()
        dkrauss_exchange = dKrauss.nth_kraus_exch
        assert_array_almost_equal(dkrauss_exchange, spkrauss_exchange)

        # Test channel output
        rho = np.array(rand_dm_ginibre(2 ** n).data.todense())
        assert_array_almost_equal(spkrauss.channel(rho), dKrauss.channel(rho))

        # Test entropy exchange
        assert_array_almost_equal(spkrauss.entropy_exchange(rho, n)[0],
                                  dKrauss.entropy_exchange(rho, n)[0])


def test_orthogonal_krauss_conditions():
    r"""Test that orthogonal kraus conds splits entropy exchange matrix and matches without it."""
    p, q = 0.2, 0.4

    k_ops = initialize_dephrasure_examples(p, q)
    dkrauss = DenseKraus(k_ops, [1, 1], 2, 3)
    dKrauss_cond = DenseKraus(k_ops, [1, 1], 2, 3, orthogonal_kraus=[2])
    for n in range(1, 4):
        dkrauss.update_kraus_operators(n)
        dKrauss_cond.update_kraus_operators(n)
        for _ in range(0, 50):
            rho = np.array(rand_dm_ginibre(2 ** n).data.todense())
            entropy_exchange = dkrauss.entropy_exchange(rho, n)
            entropy_exchange2 = dKrauss_cond.entropy_exchange(rho, n)
            assert_array_almost_equal(entropy_exchange[0],
                                      block_diag(entropy_exchange2[0], entropy_exchange2[1]))


def test_entropy_exchange_matrix_with_dephasing_channel():
    r"""Test that entropy exchange matches dephasing channel example."""
    n = 1
    for err in np.arange(0.0, 1., 0.01):
        krauss_1 = np.array([[1., 0.], [0, np.sqrt(1 - err)]])
        krauss_2 = np.array([[0, np.sqrt(err)], [0., 0]])
        krauss_ops = [krauss_1, krauss_2]

        krauss = DenseKraus(krauss_ops, [1, 1], 2, 2)
        rho = np.array(rand_dm_ginibre(2 ** n).data.todense())

        W = np.array([[np.trace(krauss_1.dot(rho).dot(krauss_1.conj().T)),
                       np.trace(krauss_1.dot(rho).dot(krauss_2.conj().T))],
                      [np.trace(krauss_2.dot(rho).dot(krauss_1.conj().T)),
                       np.trace(krauss_2.dot(rho).dot(krauss_2.conj().T))]])
        actual = krauss.entropy_exchange(rho, n)[0]
        assert_array_almost_equal(actual, W)


def test_channel_with_amplitude_damping_channel():
    r"""Test channel method of kraus operators with ampltitude damping example."""
    n = 1
    for err in np.arange(0.0, 1., 0.01):
        krauss_1 = np.array([[1., 0.], [0, np.sqrt(1 - err)]])
        krauss_2 = np.array([[0, np.sqrt(err)], [0., 0]])
        krauss_ops = [krauss_1, krauss_2]

        krauss = DenseKraus(krauss_ops, [1, 1], 2, 2)
        for _ in range(0, 10):
            rho = np.array(rand_dm_ginibre(2 ** n).data.todense())

            channel = krauss_1.dot(rho).dot(krauss_1.conj().T)
            channel += krauss_2.dot(rho).dot(krauss_2.conj().T)
            actual = krauss.channel(rho)
            assert_array_almost_equal(actual, channel)


def test_adjoint_of_a_channel():
    r"""Test the adjoint of the channel on DenseKraus and SparseKraus."""
    p1, p2, p3 = 0.1, 0.01, 0.4
    krauss_ops = initialize_pauli_examples(p1, p2, p3)
    # Dense Krauss
    dense_krauss_obj = DenseKraus(krauss_ops, [1, 1], 2, 2)
    sparse_krauss_obj = SparseKraus(krauss_ops, [1, 1], 2, 2)
    for n in range(1, 3):
        # Update Krauss Operators
        k_ops = krauss_ops.copy()
        if n != 1:
            k_ops = np.kron(np.array(k_ops), krauss_ops)
        dense_krauss_obj.update_kraus_operators(n)
        sparse_krauss_obj.update_kraus_operators(n)

        # Test on random density matrices
        for _ in range(0, 10):
            rho = np.array(rand_dm_ginibre(2 ** n).data.todense())

            desired = np.zeros((2**n, 2**n), dtype=np.complex128)
            for k in k_ops:
                desired += np.conj(k.T).dot(rho.dot(k))

            dense_chan = dense_krauss_obj.channel(rho, True)
            sparse_chan = sparse_krauss_obj.channel(rho, True)
            assert_array_almost_equal(desired, dense_chan)
            assert_array_almost_equal(desired, sparse_chan)


def test_adjoint_of_entropy_exchange():
    r"""Test the adjoint of the entropy exchange on pauli channel example."""
    p1, p2, p3 = 0.1, 0.01, 0.4
    krauss_ops = initialize_pauli_examples(p1, p2, p3)
    dense_krauss_obj = DenseKraus(krauss_ops, [1, 1], 2, 2)
    sparse_krauss_obj = SparseKraus(krauss_ops, [1, 1], 2, 2)

    # Test on random density matrices
    for n in [1, 2]:
        if n != 1:
            krauss_ops = np.kron(krauss_ops, krauss_ops)
        for _ in range(0, 50):
            dense_krauss_obj.update_kraus_operators(n)
            sparse_krauss_obj.update_kraus_operators(n)

            rho = np.array(rand_dm_ginibre(4 ** n).data.todense())

            desired = np.zeros((2 ** n, 2 ** n), dtype=np.complex128)
            for i, k1 in enumerate(krauss_ops):
                for j, k2 in enumerate(krauss_ops):
                    desired += np.conj(k2.T).dot(k1) * rho[j, i]

            dense_chan = dense_krauss_obj.entropy_exchange(rho, n, True)
            sparse_chan = sparse_krauss_obj.entropy_exchange(rho, n, True).todense()

            assert_array_almost_equal(dense_chan, sparse_chan)
            assert_array_almost_equal(desired, dense_chan)
            assert_array_almost_equal(desired, sparse_chan)

    # Test that adjoint maps of compl. positive maps are always unital
    rho = np.eye(4**n)
    desired = np.eye(2**n)
    assert_array_almost_equal(desired, sparse_krauss_obj.entropy_exchange(rho, n, True).todense())
    assert_array_almost_equal(desired, dense_krauss_obj.entropy_exchange(rho, n, True))


def test_trace_perserving():
    r"""Test trace-perserving method of kraus operators."""
    p1, p2, p3 = 0.1, 0.01, 0.4
    # Test pauli channel
    krauss_ops = initialize_pauli_examples(p1, p2, p3)
    # Dense Krauss
    krauss_obj = DenseKraus(krauss_ops, [1, 1], 2, 2)
    assert krauss_obj.is_trace_perserving()
    # Sparse Krauss Operators
    krauss_obj = SparseKraus(krauss_ops, [1, 1], 2, 2)
    assert krauss_obj.is_trace_perserving()

    # Dephrasure
    krauss_ops = initialize_dephrasure_examples(0.1, 0.2)
    krauss_obj = DenseKraus(krauss_ops, [1, 1], 2, 3)
    assert krauss_obj.is_trace_perserving()
    # Sparse Krauss Operators
    krauss_obj = SparseKraus(krauss_ops, [1, 1], 2, 3)
    assert krauss_obj.is_trace_perserving()

    # Erasure Channel
    krauss_ops = [np.sqrt(1. - p1) * np.array([[1., 0.], [0., 1.], [0., 0.]]),
                  np.sqrt(p1) * np.array([[0., 0.], [0., 0.], [1., 0.]]),
                  np.sqrt(p1) * np.array([[0., 0.], [0., 0.], [0., 1.]])]
    krauss_obj = DenseKraus(krauss_ops, [1, 1], 2, 3)
    assert krauss_obj.is_trace_perserving()

    # Test non-trace-perserving map.
    krauss_ops = [np.sqrt(1. - p1) * np.array([[1., 0.], [0., 1.], [0., 0.]]),
                  np.array([[0., 0.], [0., 0.], [1., 0.]]),
                  np.array([[0., 0.], [0., 0.], [0., 1.]])]
    krauss_obj = DenseKraus(krauss_ops, [1, 1], 2, 3)
    assert not krauss_obj.is_trace_perserving()


def test_entanglement_fidelity():
    r"""Test the average_entanglement_fidelity method of both Sparse and Dense Kraus Operators."""
    # Get krauss operators from dephrasure channel
    krauss_ops = initialize_dephrasure_examples(0.1, 0.2)
    chan = DenseKraus(krauss_ops, [1, 1], 2, 3)
    chan_sp = SparseKraus(krauss_ops, [1, 1], 2, 3)
    desired_fid = 0.
    probs = [0.25, 0.75]
    states = [np.array(rand_dm_ginibre(2).data.todense()),
              np.array(rand_dm_ginibre(2).data.todense())]
    for i, p in enumerate(probs):
        for k in krauss_ops:
            desired_fid += p * np.abs(np.trace(k.dot(states[i]))) ** 2

    assert np.abs(desired_fid - chan.average_entanglement_fidelity(probs, states))
    assert np.abs(desired_fid - chan_sp.average_entanglement_fidelity(probs, states))


def test_serial_concatenation():
    r"""Test serial concantenation of two krauss operators."""
    p, q = 0.1, 0.2
    k1 = initialize_dephrasure_examples(p, q)
    # Second kraus operators of dephasing channel.
    k2 = [np.sqrt(q) * np.array([[0., 1., ], [1., 0.]]),
          np.sqrt(1 - q) * np.array([[1., 0.], [0., -1.]])]

    # Serial Concatenation of "k1 \circ k2"
    desired = [x.dot(y) for x in k1 for y in k2]
    actual = DenseKraus.serial_concatenate(k1, k2)
    assert_array_almost_equal(np.array(desired), actual)

    # Test sparse
    actual = SparseKraus.serial_concatenate(k1, k2)
    assert_array_almost_equal(desired, actual.todense())

    # Test __add__ method
    dkraus1 = DenseKraus(k1, [1, 1], 2, 3)
    dkraus2 = DenseKraus(k2, [1, 1], 2, 2)
    dkraus3 = dkraus1 + dkraus2
    assert_array_almost_equal(np.array(desired), dkraus3.kraus_ops)

    # Test __add__ on sparse matrices.
    dkraus1 = SparseKraus(k1, [1, 1], 2, 3)
    dkraus2 = SparseKraus(k2, [1, 1], 2, 2)
    dkraus3 = dkraus1 + dkraus2
    assert_array_almost_equal(np.array(desired), dkraus3.kraus_ops.todense())

    # Test Incompatible dimensions.
    assert_raises(TypeError, DenseKraus.__add__, dkraus2, dkraus1)
    assert_raises(TypeError, DenseKraus.serial_concatenate, k2, k1)
    assert_raises(AssertionError, DenseKraus.serial_concatenate, dkraus1, dkraus2)
    assert_raises(AssertionError, DenseKraus.serial_concatenate, dkraus2, dkraus1)
    assert_raises(TypeError, SparseKraus.__add__, dkraus2, dkraus1)
    assert_raises(TypeError, SparseKraus.serial_concatenate, k2, k1)
    assert_raises(AssertionError, SparseKraus.serial_concatenate, dkraus1, dkraus2)
    assert_raises(AssertionError, SparseKraus.serial_concatenate, dkraus2, dkraus1)


def test_parallel_concatenation():
    r"""Test parallel concatenation of two kraus operators."""
    p, q = 0.1, 0.2
    k1 = initialize_dephrasure_examples(p, q)
    # Second kraus operators of dephasing channel.
    k2 = [np.sqrt(q) * np.array([[0., 1., ], [1., 0.]]),
          np.sqrt(1 - q) * np.array([[1., 0.], [0., -1.]])]

    # Test parallel concatenation method.
    desired = [np.kron(x, y) for x in k1 for y in k2]
    actual = DenseKraus.parallel_concatenate(k1, k2)
    assert_array_almost_equal(desired, actual)

    # Test parallel concatenation method across various inputs.
    actual = SparseKraus.parallel_concatenate(k1, k2)
    assert_array_almost_equal(desired, actual.todense())
    actual = SparseKraus.parallel_concatenate(np.array(k1), k2)
    assert_array_almost_equal(desired, actual.todense())
    actual = SparseKraus.parallel_concatenate(k1, np.array(k2))
    assert_array_almost_equal(desired, actual.todense())
    actual = SparseKraus.parallel_concatenate(np.array(k1), np.array(k2))
    assert_array_almost_equal(desired, actual.todense())

    # Test multiplication operator
    dkraus1 = DenseKraus(k1, [1, 1], 2, 3)
    dkraus2 = DenseKraus(k2, [1, 1], 2, 2)
    dkraus3 = dkraus1 * dkraus2
    assert_array_almost_equal(desired, dkraus3.kraus_ops)

    dkraus1 = SparseKraus(k1, [1, 1], 2, 3)
    dkraus2 = SparseKraus(k2, [1, 1], 2, 2)
    dkraus3 = dkraus1 * dkraus2
    assert_array_almost_equal(desired, dkraus3.kraus_ops.todense())
