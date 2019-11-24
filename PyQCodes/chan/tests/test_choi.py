r"""
The MIT License

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
import numpy as np
from numpy.testing import assert_raises
from qutip import rand_dm_ginibre

from PyQCodes.chan._choi import ChoiQutip
from PyQCodes.chan._kraus import DenseKraus

r"""Test file for PyQCodes._choi.py."""


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

    # Test dimensions must match the choi matrix specified.
    assert_raises(ValueError, ChoiQutip, choi_matrix, numb_qubits, 3, 3)
    assert_raises(ValueError, ChoiQutip, choi_matrix, numb_qubits, 2, 2)
    assert_raises(ValueError, ChoiQutip, choi_matrix, [1, 2], 2, 3)


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

    kraus = [np.array([[0., 1.], [1., 0.]])]


if __name__ == "__main__":
    pass
