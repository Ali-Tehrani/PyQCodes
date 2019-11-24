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
import projectq
import numpy as np
from numpy.testing import assert_raises
import itertools

from projectq import MainEngine
from projectq.ops import H, XGate, YGate, ZGate, HGate, All, Measure
from projectq.meta import Control
from PyQCodes.codes import StabilizerCode

r"""Test the StabilizerCode class from codes.py."""


def test_binary_representation():
    """Test binary representation of stabilizer codes."""
    stabs = ["ZZI", "ZIZ"]
    binary_rep = np.array([[0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 1, 0, 1]])
    code = StabilizerCode(stabs, 3, 1)
    assert np.all(np.abs(binary_rep - code.stab_bin_rep) < 1e-4)

    # Assert conversion from binary-representation to string.
    pauli_strings = StabilizerCode.binary_rep_to_pauli_str(binary_rep[0])
    assert all([pauli_strings[i] == stabs[0][i] for i in range(0, 3)])

    code = ["XXI", "XIX"]
    binary_rep = np.array([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]])
    logical = [np.array([[1, 0, 0, 0, 0, 0]]), np.array([[0, 0, 0, 1, 1, 1]])]
    stabilizer = StabilizerCode(code, 3, 1, logical_ops=logical)
    assert np.all(np.abs(binary_rep - stabilizer.stab_bin_rep) < 1e-4)


def test_two_pauli_operators_commute_with_inner_product():
    """Test two pauli operators commute or not."""
    # Anti-Commuting operators ["XXY", "ZII"]
    binary_rep_xxy = np.array([1, 1, 1, 0, 0, 1], dtype=np.int)
    binary_rep_zii = np.array([0, 0, 0, 1, 0, 0], dtype=np.int)
    actual = StabilizerCode.inner_prod(binary_rep_xxy, binary_rep_zii)
    assert actual == 1

    # Commuting Operators ["XIIX", "ZIIZ"]
    binary_rep_xiix = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int)
    binary_rep_ziiz = np.array([0, 0, 0, 0, 1, 0, 0, 1], dtype=np.int)
    actual = StabilizerCode.inner_prod(binary_rep_xiix, binary_rep_ziiz)
    assert actual == 0


def test_stabilizer_code_is_commutative_and_other_errors():
    r"""Test all stabilizers commute with one another."""
    stabilizers = ["ZZI", "ZIZ"]
    code = StabilizerCode(stabilizers, 3, 1)
    assert code._is_stabilizer_code()

    stabilizers = ["XII", "ZII"]
    assert_raises(AssertionError, StabilizerCode, stabilizers, 3, 1)
    assert_raises(AssertionError, StabilizerCode, stabilizers, 3, 2)
    assert_raises(AssertionError, StabilizerCode, stabilizers, 4, 1)
    assert_raises(AssertionError, StabilizerCode, stabilizers, 3, 1, [[1], [2]])
    logical = [np.array([[1, 0, 0, 0, 0]]), np.array([[0, 0, 0, 1, 1, 1]])]  # Remove One Column.
    assert_raises(AssertionError, StabilizerCode, stabilizers, 3, 1, logical)
    logical = [np.array([[1, 0, 0, 0, 0, 0]]), np.array([[0, 0, 0, 1, 1]])]  # Remove One Column
    assert_raises(AssertionError, StabilizerCode, stabilizers, 3, 1, logical)


# def test_generator_set_pauli_elements():
#     r"""Test that the generators are obtained."""
#     stabilizers = ["ZZ", "ZI"]
#     binary_rep = np.array([[0, 0, 1, 1],
#                            [0, 0, 1, 0]], dtype=np.int)
#     stabilizer = StabilizerCode(binary_rep, 2, 1)
#     actual = stabilizer.generator_set_pauli_elements()
#     desired = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
#     assert np.all(np.abs(actual - np.array(desired)) < 1e-5)
#
#     stabilizers = ["ZZI", "ZIZ"]
#     binary_rep = np.array([[0, 0, 0, 1, 1, 0],
#                            [0, 0, 0, 1, 0, 1]], dtype=np.int)
#     stabilizer = StabilizerCode(binary_rep, 3, 1)
#     desired = np.eye(6, dtype=np.int)
#     assert np.all(np.abs(stabilizer.generator_set_pauli_elements() - desired) < 1e-5)

# def test_normalizer():
#     r"""Test getting the normalizer of the stabilizer group."""
#     code = ["ZIZ", "ZZI"]
#     binary_rep = np.array([[0, 0, 0, 1, 1, 0],
#                            [0, 0, 0, 1, 0, 1]], dtype=np.int)
#     normalizer_elements = ["XXX", 'ZZZ']
#     # Order it so it matches the algorithm.
#     binary_rep_norm = np.array([[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
#     stabilizer = StabilizerCode(binary_rep, 3, 1)
#     actual = stabilizer.normalizer()
#     # assert np.all(np.abs(actual - binary_rep_norm) < 1e-5)
#     print(actual)
#     stab = StabilizerCode(['ZZI', 'IZZ'], ['XXX'], ['ZZZ'])
#     print([x for x in stab.normalizer()])


def test_kraus_operators_for_encoder():
    r"""Test the kraus operators for encoder."""
    # Test bit-flip code.
    code = ["ZZI", "ZIZ"]
    binary_rep = np.array([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]])
    stabilizr = StabilizerCode(binary_rep, 3, 1)
    desired = np.array([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]])
    actual = stabilizr.encode_krauss_operators()
    assert np.all(np.abs(desired - actual) < 1e-5)
    # Test that it is an isometry
    assert np.all(np.abs(actual.T.conj().dot(actual) - np.eye(2)) < 1e-5)

    # Test Phase-Flip Code.
    binary_rep = np.array([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]])
    stabilizr = StabilizerCode(binary_rep, 3, 1)
    actual = stabilizr.encode_krauss_operators(sparse=True).todense()

    bell_state1 = np.array([1., 1.]) / np.sqrt(2)
    bell_state2 = np.array([1., -1.]) / np.sqrt(2)
    # Get another basis for the code-space
    basis_vec1 = np.kron(bell_state1, np.kron(bell_state1, bell_state1))
    basis_vec2 = np.kron(bell_state2, np.kron(bell_state2, bell_state2))
    # Test that column space of two code-spaces are the same.
    coeff1 = np.linalg.lstsq(actual, basis_vec1, rcond=None)[0]
    coeff2 = np.linalg.lstsq(actual, basis_vec2, rcond=None)[0]
    assert np.all(np.abs(coeff1) > 1e-5)
    assert np.all(np.abs(coeff2) > 1e-5)
    # Test that it is an isometry
    assert np.all(np.abs(actual.T.conj().dot(actual) - np.eye(2)) < 1e-5)

    # Test shor code
    binary_rep = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0, 0, 0, 1, 0, 1],
                           [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    stabilizr = StabilizerCode(binary_rep, 9, 1)
    actual = stabilizr.encode_krauss_operators()
    state1 = np.array([1., 0, 0, 0, 0, 0, 0, 1])  / np.sqrt(2)
    state2 = np.array([1., 0, 0, 0, 0, 0, 0, -1]) / np.sqrt(2)
    basis_vec1 = np.kron(state1, np.kron(state1, state1))
    basis_vec2 = np.kron(state2, np.kron(state2, state2))
    coeff1 = np.linalg.lstsq(actual, basis_vec1, rcond=None)[0]
    coeff2 = np.linalg.lstsq(actual, basis_vec2, rcond=None)[0]
    assert np.all(np.abs(coeff1) > 1e-5)
    assert np.all(np.abs(coeff2) > 1e-5)
    # Test that it is an isometry
    assert np.all(np.abs(actual.T.conj().dot(actual) - np.eye(2)) < 1e-5)

    # Test Cat Code n = 2
    n, k = 2, 1
    binary_rep = np.array([[0, 0, 1, 1]])
    stabilizr = StabilizerCode(binary_rep, n, k)
    actual = stabilizr.encode_krauss_operators()
    desired = np.array([[1., 0.], [0., 0.], [0., 0.], [0., 1.]])
    assert np.all(np.abs(actual - desired) < 1e-4)


def test_syndrome_measurement_circuit_on_bit_flip_code():
    r"""Test the syndrome measurement circuit on bit flip code."""
    # Test bit-flip code.
    code = ["ZZI", "ZIZ"]
    binary_rep = np.array([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]])
    stabilizer = StabilizerCode(binary_rep, 3, 1)

    # Test on zero computational basis.
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(3)
    measure_result = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary_rep[0])
    measure_result2 = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary_rep[1])
    assert measure_result == 0
    assert measure_result2 == 0

    # Test measuring ZIZ on |101>.
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(3)
    XGate() | quantum_reg[0]
    XGate() | quantum_reg[2]
    measure_result = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary_rep[1])
    All(Measure) | quantum_reg
    eng.flush(deallocate_qubits=True)
    assert measure_result == 0

    # Test measuring ZIZ on |100>
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(3)
    XGate() | quantum_reg[0]  # Turn to |100>
    measure_result = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary_rep[1])
    All(Measure) | quantum_reg
    eng.flush(deallocate_qubits=True)
    assert measure_result == 1

    # Test measuring ZIZ on |010>
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(3)
    XGate() | quantum_reg[1]  # Turn to |010>
    measure_result = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary_rep[1])
    All(Measure) | quantum_reg
    eng.flush(deallocate_qubits=True)
    assert measure_result == 0


def test_binary_rep_to_pauli_mat():
    r"""Test converting from binary representation to pauli-matrices."""
    Z = np.array([[1., 0.], [0., -1.]])
    code = ["ZZI", "ZIZ"]
    binary_rep = np.array([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]])
    actual = StabilizerCode.binary_rep_to_pauli_mat(binary_rep)
    desired = [np.kron(Z, np.kron(Z, np.eye(2))),
               np.kron(Z, np.kron(np.eye(2), Z))]
    assert np.all(np.abs(np.array(desired) - np.array(actual)) < 1e-5)

    X = np.array([[0., 1.], [1., 0.]])
    code= ["XIX", "ZIZ"]
    binary_rep = np.array([[1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 1]])
    actual = StabilizerCode.binary_rep_to_pauli_mat(binary_rep, sparse=True)
    actual[0] = actual[0].todense()
    actual[1] = actual[1].todense()
    desired = [np.kron(X, np.kron(np.eye(2), X)),
               np.kron(Z, np.kron(np.eye(2), Z))]
    assert np.all(np.abs(np.array(desired) - actual) < 1e-5)


def test_gaussian_elimination_first_block_of_binary_representation():
    r"""
    Test gaussian elimination of binary representation of G1 in [G1 | G2].

    Results obtained from Gaitan - 'Quantum error correction and Fault Tolerant Quantum Computing.'
    """
    # Five qubit code : "XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)
    desired = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                        [0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 1, 1, 1]])
    output, rank = code._gaussian_elimination_first_block()
    assert np.all(np.abs(desired - output) < 1e-5)
    assert rank == 4

    # Test on [4, 2, 2] Code "XZZX", "YXXY"
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    desired = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                        [0, 1, 1, 0, 1, 1, 1, 1]])
    code = StabilizerCode(binary_rep, n, k)
    output, rank = code._gaussian_elimination_first_block()
    assert np.all(np.abs(desired - output) < 1e-5)
    assert rank == 2

    # Test on [8, 3, 3] Code
    n, k = 8, 3
    binary_rep = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

    desired = np.array([[1, 0, 0, 0, 1, 1, 1, 0,   0, 1, 0, 0, 1, 1, 0, 1],
                        [0, 1, 0, 0, 1, 1, 0, 1,   0, 0, 1, 0, 1, 0, 1, 1],
                        [0, 0, 1, 0, 1, 0, 1, 1,   0, 1, 0, 1, 1, 0, 1, 0],
                        [0, 0, 0, 1, 0, 1, 1, 1,   0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 1, 1, 1, 1, 1]])
    code = StabilizerCode(binary_rep, n, k)
    output, rank = code._gaussian_elimination_first_block()
    assert np.all(np.abs(desired - output) < 1e-5)
    assert rank == 4


def test_gaussian_elimination_second_block_of_binary_representation():
    r"""Test gaussian elimination of the second block of binary representation."""
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)
    gaussian_elimination1 = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                                      [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                                      [0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 1, 0, 1, 1, 1]])
    # output = code._gaussian_elimination_second_block(gaussian_elimination1, rank=4)

    n, k = 7, 1
    binary_rep = np.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]])

    # Test already put in standard normal form.
    gaus_eliminated = np.array([
                        [1, 0, 0, 0, 1, 1, 1,   0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0, 1, 1,   0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 0,   0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0,   1, 0, 1, 1, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0,   0, 1, 1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 0, 0, 1, 0]])
    code = StabilizerCode(binary_rep, n, k)
    output = code._gaussian_elimination_second_block(gaus_eliminated, 3)
    assert np.all(np.abs(output - gaus_eliminated) < 1e-5)


def test_standard_normal_form():
    r"""Test putting the binary representation into standard normal form."""
    # Test on [4, 2, 2] Code "XZZX", "YXXY"
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    desired = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                        [0, 1, 1, 0, 1, 1, 1, 1]])
    code = StabilizerCode(binary_rep, n, k)
    output, rank = code._standard_normal_form()
    assert np.all(np.abs(desired - output) < 1e-5)
    assert rank == 2

    # Five qubit code: "XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)
    desired = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                        [0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 1, 1, 1]])
    output, rank = code._standard_normal_form()
    assert np.all(np.abs(desired - output) < 1e-5)
    assert rank == 4

    # Test on [8, 3, 3] Code
    n, k = 8, 3
    binary_rep = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

    desired = np.array([[1, 0, 0, 0, 1, 1, 1, 0,   0, 1, 0, 0, 1, 1, 0, 1],
                        [0, 1, 0, 0, 1, 1, 0, 1,   0, 0, 1, 0, 1, 0, 1, 1],
                        [0, 0, 1, 0, 1, 0, 1, 1,   0, 1, 0, 1, 1, 0, 1, 0],
                        [0, 0, 0, 1, 0, 1, 1, 1,   0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 1, 1, 1, 1, 1]])
    code = StabilizerCode(binary_rep, n, k)
    output, rank = code._standard_normal_form()
    assert np.all(np.abs(desired - output) < 1e-5)
    assert rank == 4


def test_matrix_blocks_standard_form():
    r"""Test obtaining the matrix blocks from the standard form."""
    # Test on [5, 1] Code, from Gaitan book (pg 120)
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)
    standard_form = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                              [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                              [0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 0, 1, 1, 1]])
    # Desired blocks
    b = np.array([[1, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 1, 1]])
    a2 = np.array([[1], [1], [1], [1]])
    c1 = []
    e = []
    c2 = np.array([[1], [0], [0], [1]])
    rank = 4
    actual_a2, actual_e, actual_c1, actual_c2 = code._matrix_blocks_standard_form(standard_form,
                                                                                  rank)
    assert np.all(np.abs(actual_a2 - a2) < 1e-5)
    assert np.all(np.abs(actual_e - e) < 1e-5)
    assert np.all(np.abs(actual_c1 - c1) < 1e-5)
    assert np.all(np.abs(actual_c2 - c2) < 1e-5)

    # Test on [8, 3] Code from Gaitan book pg 132
    n, k = 8, 3
    binary_rep = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)
    standard_form = np.array([[1, 0, 0, 0, 1, 1, 1, 0,   0, 1, 0, 0,  1, 1, 0, 1],
                              [0, 1, 0, 0, 1, 1, 0, 1,   0, 0, 1, 0,  1, 0, 1, 1],
                              [0, 0, 1, 0, 1, 0, 1, 1,   0, 1, 0, 1,  1, 0, 1, 0],
                              [0, 0, 0, 1, 0, 1, 1, 1,   0, 0, 1, 1,  1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 1,  1, 1, 1, 1]])
    # Desired blocks
    a2 = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    e = np.array([[1, 1, 1]])
    c1 = np.array([[1], [1], [1], [1]])
    c2 = np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0]])
    rank = 4
    actual_a2, actual_e, actual_c1, actual_c2 = code._matrix_blocks_standard_form(standard_form,
                                                                                  rank)
    assert np.all(np.abs(actual_a2 - a2) < 1e-5)
    assert np.all(np.abs(actual_e - e) < 1e-5)
    assert np.all(np.abs(actual_c1 - c1) < 1e-5)
    assert np.all(np.abs(actual_c2 - c2) < 1e-5)

    # Test [4, 2] Code from Gaitan book pg (131)
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    standard_form = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                              [0, 1, 1, 0, 1, 1, 1, 1]])
    code = StabilizerCode(binary_rep, n, k)

    rank = 2
    a2 = np.array([[0, 1], [1, 0]])
    e = []
    c1 = []
    c2 = np.array([[1, 0], [1, 1]])
    actual_a2, actual_e, actual_c1, actual_c2 = code._matrix_blocks_standard_form(standard_form,
                                                                                  rank)
    assert np.all(np.abs(actual_a2 - a2) < 1e-5)
    assert actual_e.size == 0
    assert np.all(np.abs(actual_c1 - c1) < 1e-5)
    assert np.all(np.abs(actual_c2 - c2) < 1e-5)


def test_logical_operators():
    r"""Test obtaining the logical operators from the standard normal form."""
    # Test on [5, 1] Code, from Gaitan book (pg 120)
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)
    actual_X, actual_Z = code.logical_operators()
    desired_X = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 1, 0]])
    desired_Z = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    assert np.all(np.abs(desired_X - actual_X) < 1e-5)
    assert np.all(np.abs(desired_Z - actual_Z) < 1e-5)

    # Test [4, 2] Code from Gaitan book pg (131)
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)
    actual_X, actual_Z = code.logical_operators()

    desired_X = np.array([[0, 0, 1, 0, 1, 1, 0, 0],
                          [0, 0, 0, 1, 0, 1, 0, 0]])
    desired_Z = np.array([[0, 0, 0, 0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 1, 0, 0, 1]])
    assert np.all(np.abs(desired_X - actual_X) < 1e-5)
    assert np.all(np.abs(desired_Z - actual_Z) < 1e-5)

    # Test on [8, 3] Code from Gaitan book pg 132
    n, k = 8, 3
    binary_rep = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)
    actual_X, actual_Z = code.logical_operators()
    desired_X = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]])
    desired_Z = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]])
    assert np.all(np.abs(desired_X - actual_X) < 1e-5)
    assert np.all(np.abs(desired_Z - actual_Z) < 1e-5)


def encoding_circuit_containment(code, n, k):
    r"""This is a helper function on checking if the encoding circuit maps the right code-words
    to the right code-words."""
    basis = code.encode_krauss_operators()

    # Get all computational basis.
    lst = list(itertools.product([0, 1], repeat=n))

    # Test on the |0..0> k-qubit un-encoded maps to the right-code space.
    initial_tries = [[0] * k, [1] * k]
    if k != 1: # Add extra one that is non-trivial.
        not_found = True
        while not_found :
            guess = np.random.randint(0, 2, size=(k,))
            if not any((guess[:] == initial_tries).all(1)):
                initial_tries += [guess]
                not_found = False

    # Start testing on specific encoding on start-vectors.
    for start in initial_tries:
        already_done = []
        print("start", start)
        for numb_tries in range(0, 200):
            # Create engine, create register, apply encoding circuit and measure it.
            from projectq.setups import linear
            engine_list2 = linear.get_engine_list(num_qubits=5, cyclic=False,
                                                  one_qubit_gates="any",
                                                  two_qubit_gates="any")
            eng = projectq.MainEngine(engine_list=engine_list2)

            register = eng.allocate_qureg(n)
            eng.flush()
            code.encoding_circuit(eng, register, start)

            All(Measure) | register
            eng.flush()
            result = tuple([int(x) for x in register])

            if numb_tries == 0:
                # Find the right-column space.
                for j in range(0, 2 ** k):
                    basis1 = np.array(lst)[basis[:, j] != 0]
                    if any((basis1[:] == list(result)).all(1)) and j not in already_done:
                        already_done.append(j)
                        break
                else:
                    # If the loop finishes properly, then the test failed.
                    raise ValueError("Test failed, couldn't find encoded basis in the code-space.")
            assert any((basis1[:] == list(result)).all(1))


def test_encoding_circuit_5_1_code_containment():
    r"""Test the encoding circuit on the [5,1] code that it maps to the right code-space."""
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)

    encoding_circuit_containment(code, n, k)


def test_encoding_circuit_4_2_code_containment():
    r"""Test the encoding cirucit on [4, 2] code that it maps to the right code-space."""
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)
    encoding_circuit_containment(code, n, k)


def test_encoding_circuit_on_dephasing_containment():
    r"""Test the probabilies of the dephasing error code has the right probabilities."""
    n, k = 3, 1
    binary_rep = np.array([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]])
    code = StabilizerCode(binary_rep, 3, 1)

    # Create engine, create register, apply encoding circuit and measure it.
    eng = projectq.MainEngine()
    register = eng.allocate_qureg(n)
    code.encoding_circuit(eng, register, [1])

    # Check on |001>, |010>, |100>, |111>
    eng.flush()
    prob1 = eng.backend.get_probability("001", register)
    prob2 = eng.backend.get_probability("010", register)
    prob3 = eng.backend.get_probability("100", register)
    prob4 = eng.backend.get_probability("111", register)
    assert np.all(np.abs(np.array([prob1, prob2, prob3, prob4]) - np.array([0.25] * 4)) < 1e-4)
    eng.flush()
    All(Measure) | register

    # Check on |000>, |011>, |101>, |110>
    eng = projectq.MainEngine()
    register = eng.allocate_qureg(n)
    code.encoding_circuit(eng, register, [0])
    eng.flush()
    prob1 = eng.backend.get_probability("000", register)
    prob2 = eng.backend.get_probability("011", register)
    prob3 = eng.backend.get_probability("101", register)
    prob4 = eng.backend.get_probability("110", register)
    assert np.all(np.abs(np.array([prob1, prob2, prob3, prob4]) - np.array([0.25] * 4)) < 1e-4)
    eng.flush()
    All(Measure) | register


def test_encoding_circuit_on_shor_code_containment():
    r"""FIXME: THIS DOESN"T WORK."""
    n, k = 9, 1
    binary_rep = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0,   1, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 1, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 1, 0, 1],
                           [1, 1, 1, 1, 1, 1, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0, 1, 1, 1,   0, 0, 0, 0, 0, 0, 0, 0, 0]])
    code = StabilizerCode(binary_rep, 9, 1)
    basis = code.encode_krauss_operators()

    all_basis = list(itertools.product([0, 1], repeat=n))

    # print(basis[:, 0])
    print(code.normal_form)
    basis_sets = ["000000000", "000000111", "000111000", "000111111",
                  "111000000", "111000111", "111111000", "111111111"]
    prob_amplitude = [1. / (2. * np.sqrt(2))]
    for i, x in enumerate(basis[:, 0]):
        eng = projectq.MainEngine()

        register = eng.allocate_qureg(n)
        eng.flush()
        code.encoding_circuit(eng, register, [0])
        eng.flush()
        ordering, amplitudes = eng.backend.cheat()
        print(ordering)
        print(amplitudes)


        All(Measure) | register
        measured = [int(x) for x in register]
        measured_str = "".join(str(k) for k in measured)
        eng.flush(deallocate_qubits=True)
        print(measured_str)
        print(basis[:, 0] - amplitudes)
        assert measured_str in basis_sets

        # # Get basis element
        # basis_element = all_basis[i]
        # string_basis = "".join(str(k) for k in basis_element)
        # eng.flush()
        # prob = eng.backend.get_amplitude(string_basis, register)


def test_encoding_circuit_8_3_code_containment():
    r"""Test the encoding circuit on the [8, 3] code that it maps to the right code-space.

    FIXME: This doesn't work.
    """
    n, k = 8, 3
    binary_rep = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)
    encoding_circuit_containment(code, n, k)

    # Test assertion is raised if state doesn't match k.
    eng = projectq.MainEngine()
    register = eng.allocate_qureg(8)
    assert_raises(AssertionError, code.encoding_circuit, eng, register, [0])


def test_8_3_code_from_book():
    r"""Test the 8_3 code straight from the book.

    FIXME: This doesn't work.
    """

    def apply_code_from_book(eng, register):
        HGate() | register[0]
        with Control(eng, register[0]):
            # QubitOperator('Y4', -1.j) | register
            # QubitOperator('Y5', -1.j) | register
            YGate() | register[4]
            YGate() | register[5]
            XGate() | register[6]
            ZGate() | register[7]

        HGate() | register[1]
        with Control(eng, register[1]):
            # QubitOperator('Y4', -1.j) | register
            YGate() | register[4]
            XGate() | register[5]
            ZGate() | register[6]
            # QubitOperator('Y7', -1.j) | register
            YGate() | register[7]

        HGate() | register[2]
        with Control(eng, register[2]):
            ZGate() | register[1]
            # QubitOperator('Y4', -1.j) | register
            # QubitOperator('Y6', -1.j) | register
            YGate() | register[4]
            YGate() | register[6]
            XGate() | register[7]

        HGate() | register[3]
        with Control(eng, register[3]):
            ZGate() | register[2]
            ZGate() | register[4]
            # QubitOperator('Y5', -1.j) | register
            YGate() | register[5]
            XGate() | register[6]
            XGate() | register[7]

    n, k = 8, 3
    binary_rep = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)
    basis = code.encode_krauss_operators()

    # Get all computational basis.
    n, k = 8, 3
    lst = list(itertools.product([0, 1], repeat=n))

    # Start testing on specific encoding on start-vectors.
    already_done = []
    for numb_tries in range(0, 200):
        # Create engine, create register, apply encoding circuit and measure it.
        eng = projectq.MainEngine()
        register = eng.allocate_qureg(n)
        eng.flush()
        ordering, amplitudes = eng.backend.cheat()
        print("pre", ordering)
        apply_code_from_book(eng, register)
        eng.flush()
        ordering, amplitudes = eng.backend.cheat()
        print(ordering)
        print(amplitudes)
        All(Measure) | register
        print("prob ", eng.backend.get_amplitude("11010100", register))
        ordering, amplitudes = eng.backend.cheat()
        print(ordering)
        eng.flush()
        result = tuple([int(x) for x in register])
        print("measred ", "".join([str(x) for x in result]))

        del register
        if numb_tries == 0:
            # Find the right-column space.
            for j in range(0, 2 ** k):
                basis1 = np.array(lst)[basis[:, j] != 0]
                if any((basis1[:] == list(result)).all(1)) and j not in already_done:
                    already_done.append(j)
                    break
            else:
                # If the loop finishes properly, then the test failed.
                raise ValueError("Test failed, couldn't find encoded basis in the code-space.")
        print("j ", j)
        print(result)
        print((basis1[:] == list(result)).all(1))
        print(basis1)
        assert any((basis1[:] == list(result)).all(1))


def test_probabilities_of_encoding_circuit_against_encoding_operator_4_2_code():
    r"""Test the probabilities of encoding circuit against the encoding operator [4, 2] code.

    This test depends on "encode_kraus_operator" working properly.
    """
    # Test on [4, 2] Code.
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)

    # Kraus Operator for the encoding channel.
    kraus = code.encode_krauss_operators()

    already_found = []
    for state in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        eng = projectq.MainEngine()
        register = eng.allocate_qureg(n)
        eng.flush() # Fix the ordering of the register.
        code.encoding_circuit(eng, register, state=state)

        # Get Probabilities
        eng.flush()  # Need to flush in order to cheat.
        order, amplitudes = eng.backend.cheat()
        probabilites = np.abs(np.array(amplitudes))**2.

        # Find column vector such that theyr'e the same
        found = False
        for i in range(0, 2**k):   # Go throough each column
            if i not in already_found and not found:
                # If the probabilities match, then it was found.
                if np.all(np.abs(probabilites - kraus[:, i]**2.) < 1e-5):
                    already_found.append(i)
                    found = True
        assert found, "Did not found a column-vector that matches the probabilities. Test failed"

        All(Measure) | register
    assert len(already_found) == 2**k


def test_probabilities_of_encoding_circuit_against_encoding_operator_5_1_code():
    r"""Test the probabilities of encoding circuit against the encoding operator [5, 1] code.

    This test depends on "encode_kraus_operator" working properly.
    """
    # Test on [4, 2] Code.
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)

    # Isometry Operator for the encoding channel.
    kraus = code.encode_krauss_operators()

    already_found = []
    for state in [[0], [1]]:
        # Construct the circuit and run it.
        eng = projectq.MainEngine()
        register = eng.allocate_qureg(n)
        eng.flush() # Fix the ordering of the register.
        code.encoding_circuit(eng, register, state=state)

        # Get Probabilities
        eng.flush()  # Need to flush in order to cheat.
        order, amplitudes = eng.backend.cheat()
        probabilites = np.array(np.abs(amplitudes))**2.

        # Find column vector such that they're the same.
        found = False
        for i in range(0, 2**k):   # Go through each column
            if i not in already_found and not found:
                # If the probabilities match, then it was found.
                if np.all(np.abs(probabilites - kraus[:, i]**2.) < 1e-5):
                    already_found.append(i)
                    found = True
        assert found, "Did not found a column-vector that matches the probabilities. Test failed"

        All(Measure) | register
    assert len(already_found) == 2**k


def test_decoding_circuit_with_encoding_circuit_perserves_state_5_1_code():
    r"""Test the decoding circuit applied after encoding circuit perserves the state [5, 1] code."""
    # Test on [5, 1] Code.
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)

    # Construct the circuit and run it.
    for state in [[0], [1]]:
        for i in range(0, 100):  # Do this 100 times.
            eng = projectq.MainEngine()
            register = eng.allocate_qureg(n)
            eng.flush()  # Fix the ordering of the register.
            code.encoding_circuit(eng, register, state=state)
            if i % 2 == 0:
                total_register = code.decoding_circuit(eng, register, add_ancilla_bits=True)
            else:
                # Test the deallocation of n-qubits to increase coverage.
                total_register = code.decoding_circuit(eng, register, add_ancilla_bits=True,
                                                       deallocate_nqubits=True)
            All(Measure) | total_register

            result = int(total_register[-1])
            assert result == state[0]


def test_decoding_circuit_with_encoding_circuit_perserves_state_4_2_code():
    r"""Test the decoding circuit applied after encoding circuit perserves the state [4, 2] code."""
    # Test on [4, 2] Code.
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)

    # Construct the circuit and run it.
    for state in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        # Do this 100 times.
        for _ in range(0, 100):
            eng = projectq.MainEngine()
            register = eng.allocate_qureg(n + k)  # Add the extra ancilla bits already.
            eng.flush()  # Fix the ordering of the register.
            code.encoding_circuit(eng, register[:n], state=state)
            total_register = code.decoding_circuit(eng, register, add_ancilla_bits=False,
                                                   deallocate_nqubits=True)
            All(Measure) | total_register

            result = [int(total_register[-2]), int(total_register[-1])]
            assert len(total_register) == 2
            assert result[0] == state[0]
            assert result[1] == state[1]


def test_code_concatenation_shor_code():
    r"""Test code concatenation for the shor code."""
    # Phase - Flip
    binary_rep = np.array([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]])
    stabilizer1 = StabilizerCode(binary_rep, 3, 1)

    # Bit - Flip
    binary_rep = np.array([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]])
    stabilizer2 = StabilizerCode(binary_rep, 3, 1)

    concate = StabilizerCode.concatenate_codes(stabilizer1, stabilizer2)
    desired_bin = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0, 0, 0, 1, 0, 1],
                            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    assert np.all(np.abs(desired_bin - concate.stab_bin_rep) < 1e-4)

    # Test out the kraus operators
    kraus1 = stabilizer1.encode_krauss_operators()
    kraus2 = stabilizer2.encode_krauss_operators()
    kraus2 = np.kron(kraus2, np.kron(kraus2, kraus2))

    actual = kraus2.dot(kraus1)
    kraus3 = concate.encode_krauss_operators()

    # Assert that their singular values are the same
    singular1 = np.linalg.svd(kraus3)
    singular2 = np.linalg.svd(actual)
    assert np.all(np.abs(singular1[1] - singular2[1]) < 1e-4)


def test_code_concatenation_4_2_code_with_itself():
    r"""Test the code concatenation of the [4, 2] code with itself."""
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    code1 = StabilizerCode(binary_rep, n, k)
    code2 = StabilizerCode(binary_rep, n, k)

    # Set up logical operators so that it matches the Gaitan's book.
    code2._logical_x = np.array([[1, 0, 1, 1, 0, 0, 1, 1],
                                 [1, 0, 1, 0, 0, 0, 0, 1]])
    code2._logical_z = np.array([[1, 0, 1, 0, 1, 1, 1, 0],
                                 [0, 1, 0, 0, 0, 0, 1, 1]])

    desired = np.array([[1, 0, 0, 1, 0, 0, 0, 0,   0, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0,   1, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 1,   0, 0, 0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1,   0, 0, 0, 0, 1, 0, 0, 1],
                        [1, 1, 1, 1, 0, 0, 0, 0,   0, 0, 0, 0, 1, 1, 1, 1],
                        [1, 0, 1, 1, 0, 1, 0, 1,   1, 1, 0, 0, 0, 0, 0, 1]])
    code = StabilizerCode.concatenate_codes(code1, code2)
    assert np.all(np.abs(desired - code.stab_bin_rep) < 1e-4)
    #  Test out the kraus operators
    kraus1 = code1.encode_krauss_operators()
    kraus2 = code2.encode_krauss_operators()
    kraus2 = np.kron(kraus2, kraus2)

    actual = kraus2.dot(kraus1)
    kraus3 = code.encode_krauss_operators()

    # Assert that their singular values are the same
    singular1 = np.linalg.svd(kraus3)
    singular2 = np.linalg.svd(actual)
    assert np.all(np.abs(singular1[1] - singular2[1]) < 1e-4)


def test_code_concatenation_5_1_code_with_itself():
    r"""Test code concatenating the [5, 1] code with itself."""
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code1 = StabilizerCode(binary_rep, n, k)
    code2 = StabilizerCode(binary_rep, n, k)
    # Set up the logical operators so it matches what is in the book.
    code2._logical_x = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    code2._logical_z = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    code = StabilizerCode.concatenate_codes(code1, code2)

    # Test the first four rows.
    desired_x = np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0],
                          [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0],
                          [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0],
                          [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0]])
    desired_z = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0],
                          [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0],
                          [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0]])
    desired = np.hstack((desired_x, desired_z))
    assert np.all(np.abs(code.stab_bin_rep[:4, :] - desired) < 1e-4)

    # Test the last four rows.
    desired_x = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
                           0, 0],
                          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                           1, 1],
                          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0],
                          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
                           0, 0]])
    desired_z = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                           0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1],
                          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                           1, 1]])
    desired = np.hstack((desired_x, desired_z))
    assert np.all(np.abs(code.stab_bin_rep[20:, :] - desired) < 1e-4)


def test_kraus_operators_correcting_errors_simple_example():
    r"""Test the ability to decode the kraus operators."""
    # Suppose the errors is the bit-flip plus dephasing map.
    I = np.eye(2)
    X = np.array([[0., 1.], [1., 0.]])
    Z = np.array([[1., 0.], [0., -1.]])
    errors = [np.sqrt(0.8) * np.kron(I, np.kron(I, I)),
              np.sqrt(0.1) * np.kron(np.kron(X, I), I),
              np.sqrt(0.1) * np.kron(I, np.kron(I, Z))]

    # Easy case.
    stabilizers = ["XII", "IIZ"]
    code = StabilizerCode(stabilizers, 3, 1)
    actual = code.kraus_operators_correcting_errors(errors, False)
    desired = [np.sqrt(0.8) * np.kron(I, np.kron(I, I)),
               np.sqrt(0.1) * np.kron(np.kron(X, I), I),
               np.sqrt(0.1) * np.kron(I, np.kron(I, Z))]

    for i in range(0, 3):
        assert np.all(np.abs(actual[i] - desired[i]) < 1e-4)

    # Correcting one kraus operator case.
    stabilizers = ["ZII"]
    matrix_stab = np.kron(Z, np.kron(I, I))
    code = StabilizerCode(stabilizers, 3, 2)
    actual = code.kraus_operators_correcting_errors(errors, False)
    desired = [np.sqrt(0.8) * np.kron(I, np.kron(I, I)),
               matrix_stab.dot(errors[1]), errors[2]]
    for i in range(0, 3):
        assert np.all(np.abs(actual[i] - desired[i]) < 1e-4)


def test_syndrome_measurement_circuit_on_phase_flip_code():
    r"""Test the syndrome measurement circuit on phase flip code.

    This tests assumes encoding circuit is working.
    """
    code = ["XXI", "XIX"]
    binary = np.array([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]])
    stabilizer = StabilizerCode(code, 3, 1)

    # Using the encoding circuit, encode the state |0> to |+++>.
    # Then measure it using any of the stabilizers. Since it is in the eigenvector basis it
    # should be zero.
    for i in range(0, 50):
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(3)
        stabilizer.encoding_circuit(eng, quantum_reg, state=[i % 2])  # Try both states.
        if i % 2 == 0:
            measure_result = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary[0])
        else:
            # Try other stabilizer.
            measure_result = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary[1])
        All(Measure) | quantum_reg
        eng.flush()
        assert measure_result == 0

        # Try applying a random logical circuit which should still be in the code-space.
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(3)
        stabilizer.encoding_circuit(eng, quantum_reg, state=[i % 2])  # Try both states.
        random_logical = np.random.choice(["X0", "Z0", "Y0"])
        stabilizer.logical_circuit(eng, quantum_reg, random_logical)
        if i % 2 == 0:
            measure_result = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary[0])
        else:
            # Try other stabilizer.
            measure_result = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary[1])
        All(Measure) | quantum_reg
        eng.flush()
        assert measure_result == 0

    # Now encode |0> to |+++>, then use the zgate on first qubit to get |-++>.
    # This isn't in the code space hence the measuremenet result is 1 indicating they anti-commute.
    for i in range(0, 50):
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(3)
        stabilizer.encoding_circuit(eng, quantum_reg, state=[0])
        ZGate() | quantum_reg[0]
        if i % 2 == 0:
            measure_result = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary[0])
        else:
            # Try other stabilizer.
            measure_result = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary[1])
        All(Measure) | quantum_reg
        eng.flush()
        assert measure_result == 1


def test_syndrome_measurement_circuit_on_4_2_code():
    r"""Test syndrome measurement on the 4 2 code."""
    # Stabilizers are "XZZX" and "YXXY" and it has two logical X and two logical Z operators.
    n, k = 4, 2
    binary = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                       [1, 1, 1, 1, 1, 0, 0, 1]])
    code = StabilizerCode(binary, n, k)

    for i in range(0, 100):
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(4)
        random_state = [i % 2, (i + np.random.random_integers(0, 3)) % 2]
        code.encoding_circuit(eng, quantum_reg, state=random_state)

        if i % 2 == 0:
            measure_result = code.single_syndrome_measurement(eng, quantum_reg, "XZZX")
        else:
            measure_result = code.single_syndrome_measurement(eng, quantum_reg, "YXXY")
        All(Measure) | quantum_reg
        eng.flush()
        assert measure_result == 0

        # Try applying a random logical circuit which should still be in the code-space.
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(5)
        code.encoding_circuit(eng, quantum_reg[:4], state=random_state)
        random_logical = np.random.choice(["X0", "Z0", "Y0", "X1", "Z1", "Y1"])
        code.logical_circuit(eng, quantum_reg[:4], random_logical)
        if i % 2 == 0:
            measure_result = code.single_syndrome_measurement(eng, quantum_reg, binary[0])
        else:
            # Try other stabilizer.
            measure_result = code.single_syndrome_measurement(eng, quantum_reg, binary[1])
        All(Measure) | quantum_reg
        eng.flush()
        assert measure_result == 0

    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(6)
    assert_raises(TypeError, code.single_syndrome_measurement, eng, quantum_reg, binary[0])


if __name__ == "__main__":
    # test_decoding_circuit_with_encoding_circuit_perserves_state_4_2_code()
    # test_syndrome_measurement_circuit_on_4_2_code()
    # test_syndrome_measurement_circuit_on_phase_flip_code()
    # test_syndrome_measurement_circuit_on_bit_flip_code()
    pass
