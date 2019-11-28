r"""
The MIT License.

Copyright (c) 2019-Present PyQCodes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from PyQCodes.chan.channel import AnalyticQChan
from PyQCodes.chan.param import OverParam

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_raises
from qutip import rand_dm_ginibre
from scipy.linalg import block_diag

r"""
Test the file "PyQCodes.channel.py."
"""


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
    return krauss_ops


def test_channel_method_using_dephrasure():
    r"""Test the method 'channel' from 'QCorr.channel.AnalyticQChan' using dephrasure kraus ops."""
    p, q = 0.2, 0.4

    single_krauss_ops = set_up_dephrasure_conditions(p, q)
    orthogonal_krauss_indices = [2]
    for sparse in [False, True]:
        krauss_ops = single_krauss_ops.copy()
        channel = AnalyticQChan(single_krauss_ops, [1, 1], 2, 3, orthogonal_krauss_indices, sparse)
        for n in range(1, 4):
            if n != 1:
                krauss_ops = np.kron(single_krauss_ops, krauss_ops)

            # Get random density state.
            rho = np.array(rand_dm_ginibre(2**n).data.todense())

            # Test channel method using kraus perators matches 'channel.channel'
            desired = np.zeros((3**n, 3**n), dtype=np.complex128)
            for krauss in krauss_ops:
                temp = krauss.dot(rho).dot(krauss.T)
                desired += temp
            actual = channel.channel(rho, n)
            assert_array_almost_equal(actual, desired)

    # Test constructor accepts the right input.
    assert_raises(TypeError, AnalyticQChan, "asdsa", [1, 1], 2, 2)


def test_entropy_exchange_dephrasure_channel():
    r"""Test the entropy exchange of dephrasure channel using kraus operators."""
    p, q = 0.2, 0.4

    single_krauss_ops = set_up_dephrasure_conditions(p, q)
    orthogonal_krauss_indices = [2]
    # Test both sparse and non-sparse.
    for issparse in [True, False]:
        krauss_ops = single_krauss_ops.copy()
        channel = AnalyticQChan(single_krauss_ops, [1, 1], 2, 3, orthogonal_krauss_indices,
                                sparse=issparse)
        for n in range(1, 4):
            if n != 1:
                krauss_ops = np.kron(single_krauss_ops, krauss_ops)

            # Get random density state.
            rho = np.array(rand_dm_ginibre(2 ** n).data.todense())

            if issparse:
                actual = channel.entropy_exchange(rho, n)[0]
            else:
                actual = channel.entropy_exchange(rho, n)
                actual = block_diag(actual[0], actual[1])
            assert actual.shape == (4**n, 4**n)
            for i in range(0, 4**n):
                for j in range(0, 4**n):
                    desired = np.trace(krauss_ops[i].dot(rho.dot(np.conjugate(krauss_ops[j]).T)))
                    assert np.abs(actual[i, j] - desired) < 1e-10


def _analytic_solution(lamda, n, p, q):
    r"""
        This is the analytic coherent information for the dephrasure channel,
        found in the paper,
        "Dephrasure channel and superadditivity of coherent information."
    """
    if lamda == 0. or lamda == 1.:
        h_lamda = 0
    else:
        h_lamda = -lamda * np.log2(lamda) - (1. - lamda) * np.log2(1 - lamda)
    a1 = ((1. - q) ** n - q ** n) * h_lamda
    u = np.sqrt(1. - 4. * lamda * (1. - lamda) * (1 - (1. - 2. * p) ** (2. * n)))
    a2 = ((1. - q) ** n) * \
         (1. - (u * np.log2((1 + u) / (1 - u)) / 2.) - (np.log2(1. - u ** 2.) / 2.))
    return a1 - a2


def test_coherent_information_with_analytic_dephrasure():
    r""" Test coherent information versus an analytic example for dephrasure channel."""
    p, q = 0.2, 0.4

    single_krauss_ops = set_up_dephrasure_conditions(p, q)
    orthogonal_krauss_indices = [2]

    for sparse in [False, True]:
        channel = AnalyticQChan(single_krauss_ops, [1, 1], 2, 3, orthogonal_krauss_indices, sparse)
        for n in range(1, 4):
            for lam in np.arange(0.001, 1., 0.1):
                rho = np.zeros((2**n, 2**n), dtype=np.complex128)
                rho[0, 0] = lam
                rho[-1, -1] = 1. - lam

                if n == 1:
                    actual = channel.coherent_information(rho, n)
                else:
                    # Test regularized keyword.
                    actual = channel.coherent_information(rho, n, regularized=True) * float(n)
                desired = _analytic_solution(lam, n, p, q)
                assert np.abs(desired - actual) < 1e-10


def test_maxima_first_coherent_information_dephrasure():
    r""" Test maxima of first coh-info of dephrasure channel is mixed state and matches paper."""
    def optima_mixed_state_bound(p):
        # Bound for when coherent information is maximized by maximally mixed state.
        a = (1. - 2. * p - 2. * p * (1. - p) * np.log((1. - p) / p))
        return a / (2. - 4. * p - 2. * p * (1. - p) * np.log((1. - p) / p))

    desired = np.array([[0.5, 0.], [0., 0.5]])
    for p in np.arange(0.01, 0.5, 0.1):
        for q in np.arange(0., optima_mixed_state_bound(p), 0.1):
            # Set up actual results.
            single_krauss_ops = set_up_dephrasure_conditions(p, q)
            channel = AnalyticQChan(single_krauss_ops, [1, 1], 2, 3, [2])
            actual = channel.optimize_coherent(n=1, rank=2, param="overparam", maxiter=15)

            # Test optimal coherent information value.
            desired_fun = (1. - 2. * q) * (-np.log2(0.5)) - (1. - q) * \
                (-(1 - p) * np.log2(1 - p) - p * np.log2(p))
            assert np.abs(actual["optimal_val"] - desired_fun) < 1e-5

            # Test optimal density state is the maximally mixed state.
            assert_array_almost_equal(desired, actual["optimal_rho"], decimal=3)


# Very slow test.
@pytest.mark.slow
def test_optimizing_coherent_information_with_erasure_channel():
    r""" Test optimizing coherent information with an analytic example for erasure channel."""
    for err in np.arange(0.01, 1., 0.01):
        krauss_1 = np.sqrt(1 - err) * np.array([[1., 0.], [0., 1.], [0., 0.]])
        krauss_2 = np.sqrt(err) * np.array([[0., 0.], [0., 0.], [0., 1.]])
        krauss_3 = np.sqrt(err) * np.array([[0., 0.], [0., 0.], [1., 0.]])
        krauss_ops = [krauss_1, krauss_3, krauss_2]

        channel = AnalyticQChan(krauss_ops, [1, 1], 2, 3)
        actual = channel.optimize_coherent(n=1, rank=2, param="cholesky", maxiter=50)
        desired = 1. - err * 2.
        if err > 0.5:
            desired = 0.
        assert np.abs(actual["optimal_val"] - desired) < 1e-5


# Very slow test.
@pytest.mark.slow
def test_optimizing_coherent_information_with_amplitude_damping_channel():
    r""" Test optimizing coherent information with an analytic example for amplitude-damp."""
    def entropy(a):
        # Analytical result.
        log = np.log2(a)
        log[np.isinf(log)] = 0.

        log2 = np.log2(1. - a)
        log2[np.isinf(log2)] = 0.
        return -a * log - (1. - a) * log2

    for err in np.arange(0.01, 1., 0.01):
        # Kraus operators for amplitude-damping channel.
        krauss_1 = np.array([[1., 0.], [0., np.sqrt(1 - err)]], dtype=np.complex128)
        krauss_2 = np.array([[0., np.sqrt(err)], [0., 0.]], dtype=np.complex128)
        krauss_ops = [krauss_1, krauss_2]

        channel = AnalyticQChan(krauss_ops, [1, 1], 2, 2)
        actual = channel.optimize_coherent(n=1, rank=2, param="cholesky", maxiter=50)

        ss = 0.0001
        grid = np.arange(0., 1. + ss, ss)
        desired = np.max(-entropy(err * grid) + entropy((1 - err) * grid))
        if err > 0.5:
            desired = 0.
        assert np.abs(actual["optimal_val"] - desired) < 1e-3


def test_optimizing_coherent_information_bit_flip_channels_only_on_one_case():
    r"""Test optimizing once coherent information with an analytic example for bit-flip channel."""
    err = 0.1
    krauss_1 = np.array([[1., 0.], [0., 1]], dtype=np.complex128) * np.sqrt(1 - err)
    krauss_2 = np.array([[0., 1], [1., 0.]], dtype=np.complex128) * np.sqrt(err)
    krauss_ops = [krauss_1, krauss_2]
    desired = 1 + np.log2(err) * err + np.log2(1 - err) * (1 - err)

    # Kraus Operators
    for n in range(1, 3):
        channel = AnalyticQChan(krauss_ops, [1, 1], 2, 2)
        actual = channel.optimize_coherent(n=n, rank=2**n, param="cholesky", maxiter=100,
                                           regularized=True)
        assert np.abs(actual["optimal_val"] - desired) < 1e-3
    # Test downgrading a n.
    actual = channel.optimize_coherent(n=1, rank=2, param="overparam", maxiter=100)
    assert np.abs(actual["optimal_val"] - desired) < 1e-3

    # Choi matrices
    choi_mat = sum([np.outer(np.ravel(x, order="F"),
                             np.conj(np.ravel(x, order="F"))) for x in krauss_ops])
    chan = AnalyticQChan(choi_mat, [1, 1], 2, 2)
    actual = chan.optimize_coherent(n=1, rank=2, param="overparam", maxiter=100)
    assert np.abs(actual["optimal_val"] - desired) < 1e-3
    assert_raises(AssertionError, chan.optimize_coherent, 2, 2)


@pytest.mark.slow
def test_optimizing_coherent_information_bit_flip_channels():
    r"""Test optimizing coherent information with an analytic example for bit-flip channel."""
    for err in np.arange(0.01, 1., 0.01):
        krauss_1 = np.array([[1., 0.], [0., 1]], dtype=np.complex128) * np.sqrt(1 - err)
        krauss_2 = np.array([[0., 1], [1., 0.]], dtype=np.complex128) * np.sqrt(err)
        krauss_ops = [krauss_1, krauss_2]

        channel = AnalyticQChan(krauss_ops, [1, 1], 2, 2)
        actual = channel.optimize_coherent(n=1, rank=2, param="cholesky", maxiter=50)

        desired = 1 + np.log2(err) * err + np.log2(1 - err) * (1 - err)
        print(err, desired, actual["optimal_val"])
        assert np.abs(actual["optimal_val"] - desired) < 1e-3


def compare_lipschitz_slsqp_with_diffev():
    n = 3
    for p in np.arange(0.05, 0.5, 0.01):
        for q in np.arange(0.3, 0.5, 0.01):
            single_krauss_ops = set_up_dephrasure_conditions(p, q)
            orthogonal_krauss_indices = [2]

            channel = AnalyticQChan(single_krauss_ops, [1, 1], 2, 3, orthogonal_krauss_indices)

            diffev = channel.optimize_coherent(n, 2**n, "diffev", maxiter=500, lipschitz=10)
            slsqp = channel.optimize_coherent(n, 2**n, "slsqp", lipschitz=20, maxiter=100,
                                              use_pool=True)

            # Test optimal coherent information is the same inbetween both of them.
            assert np.abs(diffev["optimal_val"] - slsqp["optimal_val"]) < 1e-3


def check_two_sets_of_krauss_are_same(krauss1, krauss2, numb=1000):
    is_same = True
    chann1 = AnalyticQChan(krauss1, [1, 1], 2, 2)
    chann2 = AnalyticQChan(krauss2, [1, 1], 2, 2)
    for _ in range(0, numb):
        # Get random Rho
        rho = np.array(rand_dm_ginibre(2).data.todense())
        rho1 = chann1.channel(rho, 1)
        rho2 = chann2.channel(rho, 1)
        # Compare them
        if np.any(np.abs(rho1 - rho2) > 1e-3):
            is_same = False
            break
    return is_same


@pytest.mark.slow
def test_minimum_fidelity_over_depolarizing_channel():
    r"""Test Minimum fidelity over depolarizing channel."""
    # Example obtained from nielsen and Chaung.
    prob = np.arange(0., 1., 0.01)

    for p in prob:
        # With Kraus Operators
        I = np.sqrt(1 - p) * np.array([[1., 0.], [0., 1.]])
        X = np.sqrt(p / 3.) * np.array([[0., 1.], [1., 0.]])
        Y = np.sqrt(p / 3.) * np.array([[0., complex(0., -1.)], [complex(0., 1.), 0.]])
        Z = np.sqrt(p / 3.) * np.array([[1., 0.], [0., -1.]])

        kraus = [I, X, Y, Z]
        chan = AnalyticQChan(kraus, [1, 1], 2, 2)
        desired = np.sqrt(1 - 2. * p / 3.)
        actual = chan.optimize_fidelity(n=1)
        assert np.all(np.abs(desired - actual["optimal_val"]) < 1e-5)

        # With Choi-Matrix
        choi_mat = sum([np.outer(np.ravel(x, order="F"),
                                 np.conj(np.ravel(x, order="F"))) for x in kraus])
        chan = AnalyticQChan(choi_mat, [1, 1], 2, 2)
        actual = chan.optimize_fidelity(n=1)
        assert np.all(np.abs(desired - actual["optimal_val"]) < 1e-5)


@pytest.mark.slow
def test_minimum_fidelity_over_bit_flip():
    r"""Test Minimum fidelity over bit-flip channel."""
    # Example obtained from "Quantum Computing Explained."
    prob = np.arange(0., 1., 0.01)

    for p in prob:
        # With Kraus Operators
        I = np.sqrt(1 - p) * np.array([[1., 0.], [0., 1.]])
        X = np.sqrt(p) * np.array([[0., 1.], [1., 0.]])

        kraus = [I, X]
        chan = AnalyticQChan(kraus, [1, 1], 2, 2)
        desired = np.sqrt(1 - p)
        actual = chan.optimize_fidelity()
        assert np.all(np.abs(desired - actual) < 1e-5)

        # With Choi-Matrix
        choi_mat = sum([np.outer(np.ravel(x, order="F"),
                                 np.conj(np.ravel(x, order="F"))) for x in kraus])
        chan = AnalyticQChan(choi_mat, [1, 1], 2, 2)
        actual = chan.optimize_fidelity()
        assert np.all(np.abs(desired - actual) < 1e-5)


def test_minimum_fidelity_over_amplitude_damping():
    r"""Test Minimum fidelity over amplitude-damping channel."""
    # Example obtained from Nielsen and Chaung.
    prob = np.arange(0., 1., 0.5)

    for p in prob:
        # With Kraus Operators
        k0 = np.array([[1., 0.], [0., np.sqrt(1. - p)]])
        k1 = np.array([[0., np.sqrt(p)], [0., 0.]])

        kraus = [k0, k1]
        chan = AnalyticQChan(kraus, [1, 1], 2, 2)
        desired = np.sqrt(1 - p)

        actual = chan.optimize_fidelity(n=1, maxiter=100)
        assert np.all(np.abs(desired - actual["optimal_val"]) < 1e-3)

        # With Choi-Matrix
        choi_mat = sum([np.outer(np.ravel(x, order="F"),
                                 np.conj(np.ravel(x, order="F"))) for x in kraus])
        chan = AnalyticQChan(choi_mat, [1, 1], 2, 2)
        actual = chan.optimize_fidelity(n=1, maxiter=100)
        assert np.all(np.abs(desired - actual["optimal_val"]) < 1e-3)
        assert_raises(AssertionError, chan.optimize_fidelity, 2)


def test_minimum_fidelity_over_identity_channel():
    k0 = np.array([[1., 0.], [0., 1.]])
    chan = AnalyticQChan([k0], [1, 1], 2, 2)
    for n in range(1, 3):
        actual = chan.optimize_fidelity(n, maxiter=100)
        assert np.all(np.abs(1. - actual["optimal_val"]) < 1e-3)


# Slow test
@pytest.mark.slow
def test_optimizing_coherent_information_erasure_using_choi_matrix():
    r"""Test optimizing coherent information with an analytic example for erasure channel."""
    for err in np.arange(0.01, 1., 0.01):
        krauss_1 = np.sqrt(1 - err) * np.array([[1., 0.], [0., 1.], [0., 0.]])
        krauss_2 = np.sqrt(err) * np.array([[0., 0.], [0., 0.], [0., 1.]])
        krauss_3 = np.sqrt(err) * np.array([[0., 0.], [0., 0.], [1., 0.]])
        krauss_ops = [krauss_1, krauss_3, krauss_2]
        choi =  sum([np.outer(np.ravel(x, order="F"),
                                 np.conj(np.ravel(x, order="F"))) for x in krauss_ops])

        channel = AnalyticQChan(choi, numb_qubits=[1, 1], dim_in=2, dim_out=3)
        actual = channel.optimize_coherent(1, 2, param="cholesky", maxiter=50)

        desired = 1. - err * 2.
        if err > 0.5:
            desired = 0.
        assert np.abs(actual["optimal_val"] - desired) < 1e-3


# Very slow test
@pytest.mark.slow
def test_optimizing_coherent_information_dephrasure_using_choi_matrix():
    r"""Test maxima of coherent information of dephrasure based on choi matrix."""
    def optima_mixed_state_bound(p):
        # Bound where maximally mixed state is optimal.
        a = (1. - 2. * p - 2. * p * (1. - p) * np.log((1. - p) / p))
        return a / (2. - 4. * p - 2. * p * (1. - p) * np.log((1. - p) / p))

    desired = np.array([[0.5, 0.], [0., 0.5]]) # Maximally mixed state.
    n = 1
    for p in np.arange(0.01, 0.5, 0.05):
        for q in np.arange(0., optima_mixed_state_bound(p), 0.1):
            krauss_ops = set_up_dephrasure_conditions(p, q)
            choi = sum([np.outer(np.ravel(x, order="F"),
                                 np.conj(np.ravel(x, order="F"))) for x in krauss_ops])

            channel = AnalyticQChan(choi, [1, 1], 2, 3)
            actual = channel.optimize_coherent(n, 2, param=OverParam, maxiter=15)

            # Test optimal coherent information value is the same as analytical example.
            desired_fun = (1. - 2. * q) * (-np.log2(0.5)) - (1. - q) * \
                (-(1 - p) * np.log2(1 - p) - p * np.log2(p))
            assert np.abs(actual["optimal_val"] - desired_fun) < 1e-3

            # Test optimal rho is the same as analytical example
            actual = actual["optimal_rho"]
            assert_array_almost_equal(desired, actual, decimal=3)


def test_qubit_condition():
    r"""Test that qubit channels are recognized."""
    # Not a qubit channel.
    kraus = set_up_dephrasure_conditions(0.1, 0.2)
    not_qubit_chan = AnalyticQChan(kraus, [1, 1], 2, 3)
    assert not not_qubit_chan._is_qubit_channel()

    # Qubit Channel.
    err = 0.01
    krauss_1 = np.sqrt(1 - err) * np.array([[1., 0.], [0., 1.]])
    krauss_2 = np.sqrt(err) * np.array([[0., 1.], [1., 0.]])
    qubit_chan = AnalyticQChan([krauss_1, krauss_2], [1, 1], 2, 2)
    assert qubit_chan._is_qubit_channel()


def test_adding_channels_together():
    r"""Test adding channels together."""
    # Identity channel
    kraus0 = [np.eye(2)]
    chan1 = AnalyticQChan(kraus0, [1, 1], 2, 2)

    # Dephrasure Channel
    kraus = set_up_dephrasure_conditions(0.1, 0.2)
    chan2 = AnalyticQChan(kraus, [1, 1], 2, 3)

    # Raises error since they don't match.
    assert_raises(TypeError, chan1.__add__, chan2)

    # Identity channel first then
    new_chan = chan2 + chan1
    desired_kraus = [x.dot(kraus0[0]) for x in kraus]
    assert np.all(np.abs(new_chan.kraus - np.array(desired_kraus)) < 1e-4)

    # Test Sparse
    chan2 = AnalyticQChan(kraus, [1, 1], 2, 3, sparse=True)
    new_chan = chan2 + chan1
    desired_kraus = [x.dot(kraus0[0]) for x in kraus]
    assert np.all(np.abs(new_chan.kraus - np.array(desired_kraus)) < 1e-4)
    assert new_chan.sparse


def test_multipling_channels_together():
    r"""Test multiplying channels together."""
    # Identity channel
    kraus0 = [np.eye(2)]
    chan1 = AnalyticQChan(kraus0, [1, 1], 2, 2)

    # Dephrasure Channel
    kraus = set_up_dephrasure_conditions(0.1, 0.2)
    chan2 = AnalyticQChan(kraus, [1, 1], 2, 3)

    new_chan = chan1 * chan2
    desired_kraus = [np.kron(kraus0[0], x) for x in kraus]
    assert np.all(np.abs(new_chan.kraus - np.array(desired_kraus)) < 1e-4)

    new_chan = chan2 * chan1
    desired_kraus = [np.kron(x, kraus0[0]) for x in kraus]
    assert np.all(np.abs(new_chan.kraus - np.array(desired_kraus)) < 1e-4)

    # Test Sparse
    chan2 = AnalyticQChan(kraus, [1, 1], 2, 3, sparse=True)
    new_chan = chan2 * chan1
    desired_kraus = [np.kron(x, kraus0[0]) for x in kraus]
    assert np.all(np.abs(new_chan.kraus - np.array(desired_kraus)) < 1e-4)
    assert new_chan.sparse
