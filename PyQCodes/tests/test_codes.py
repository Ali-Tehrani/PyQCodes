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

from projectq import MainEngine
from projectq.ops import XGate, YGate, ZGate, HGate, All, Measure, QubitOperator
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
    # Stabilizer Codes Commute with it.
    matrices = StabilizerCode.binary_rep_to_pauli_mat(binary_rep)
    for mat in matrices:
        assert np.all(np.abs(mat.dot(actual) - actual) < 1e-4)

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
    # Stabilizer Codes Commute with it.
    matrices = StabilizerCode.binary_rep_to_pauli_mat(binary_rep)
    for mat in matrices:
        assert np.all(np.abs(mat.dot(actual) - actual) < 1e-4)

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


def test_logical_operators_from_standard_normal_form():
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


def test_decoding_circuit_with_encoding_circuit_perserves_state_shor_code():
    r"""Test the decoding circuit applied after encoding circuit perserves the state shor code."""
    # Test on [4, 2] Code.
    n, k = 9, 1
    binary_rep = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0, 0, 0, 1, 0, 1],
                           [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    code = StabilizerCode(binary_rep, n, k)

    # Construct the circuit and run it.
    for state in [[0], [1]]:
        # Do this 100 times.
        for _ in range(0, 50):
            eng = projectq.MainEngine()
            register = eng.allocate_qureg(n + k)  # Add the extra ancilla bits already.
            eng.flush()  # Fix the ordering of the register.
            code.encoding_circuit(eng, register[:n], state=state)
            total_register = code.decoding_circuit(eng, register, add_ancilla_bits=False,
                                                   deallocate_nqubits=True)
            All(Measure) | total_register

            result = int(total_register[-1])
            assert len(total_register) == 1
            assert result == state[0]


def test_decoding_circuit_with_encoding_circuit_perserves_state_8_3_code():
    r"""Test the decoding circuit applied after encoding circuit perserves the state [8, 3] code."""
    # Test on [8, 3] Code.
    n, k = 8, 3
    binary_rep = np.array([
                           [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],])
    code = StabilizerCode(binary_rep, n, k)

    # Construct the circuit and run it.
    for state in [[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1]]:
        # Do this 100 times.
        for _ in range(0, 100):
            eng = projectq.MainEngine()
            register = eng.allocate_qureg(n + k)  # Add the extra ancilla bits already.
            eng.flush()  # Fix the ordering of the register.
            code.encoding_circuit(eng, register[:n], state=state)
            total_register = code.decoding_circuit(eng, register, add_ancilla_bits=False,
                                                   deallocate_nqubits=True)
            All(Measure) | total_register

            result = [int(total_register[-3]), int(total_register[-2]), int(total_register[-1])]
            assert len(total_register) == 3
            assert result[0] == state[0]
            assert result[1] == state[1]
            assert result[2] == state[2]


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
        wave_function_before = np.array(eng.backend.cheat()[1])
        if i % 2 == 0:
            measure_result = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary[0])
        else:
            # Try other stabilizer.
            measure_result = stabilizer.single_syndrome_measurement(eng, quantum_reg, binary[1])
        wave_function_after = np.array(eng.backend.cheat()[1])
        All(Measure) | quantum_reg
        eng.flush()
        assert np.all(np.abs(wave_function_before - wave_function_after) < 1e-4)
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
        eng.flush()

        random_state = [i % 2, (i + np.random.random_integers(0, 3)) % 2]
        code.encoding_circuit(eng, quantum_reg, state=random_state)
        wave_func_bef = np.array(eng.backend.cheat()[1])
        if i % 2 == 0:
            measure_result = code.single_syndrome_measurement(eng, quantum_reg, code.normal_form[0])
        else:
            measure_result = code.single_syndrome_measurement(eng, quantum_reg, code.normal_form[1])
        wave_func_af = np.array(eng.backend.cheat()[1])
        assert np.all(np.abs(wave_func_bef - wave_func_af) < 1e-4)
        eng.flush()
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
            measure_result = code.single_syndrome_measurement(eng, quantum_reg, code.normal_form[0])
        else:
            # Try other stabilizer too .
            measure_result = code.single_syndrome_measurement(eng, quantum_reg, code.normal_form[1])
        All(Measure) | quantum_reg
        eng.flush()
        assert measure_result == 0

        # Test applying an error operator that anti-commutes.
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(4)
        eng.flush()
        code.encoding_circuit(eng, quantum_reg[:4], state=random_state)
        YGate() | quantum_reg[1]
        measure_result = code.single_syndrome_measurement(eng, quantum_reg, binary[0])
        All(Measure) | quantum_reg
        eng.flush()
        assert measure_result == 1, "Error should anticommute."

        # Test applying an error operator that anti-commutes but measure with normal form instead.
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(4)
        eng.flush()
        code.encoding_circuit(eng, quantum_reg[:4], state=random_state)
        YGate() | quantum_reg[1]
        measure_result = code.single_syndrome_measurement(eng, quantum_reg, code.normal_form[0])
        All(Measure) | quantum_reg
        eng.flush()
        assert measure_result == 1, "Error should anticommute."

    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(6)
    assert_raises(TypeError, code.single_syndrome_measurement, eng, quantum_reg, binary[0])


def test_applying_circuit_stabilizers_on_bitflip_code():
    r"""Test applying the stabilizers correctly.

    This tests assumes encoding circuit is working correctly.
    """
    code = ["ZZI", "ZIZ"]
    binary = np.array([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]])
    stabilizer = StabilizerCode(code, 3, 1)

    # Apply it 10 times with different states.
    for j in range(0, 10):
        for i in range(0, 2):  # Go through each Stabilizer.
            eng = MainEngine()
            quantum_reg = eng.allocate_qureg(3)
            stabilizer.encoding_circuit(eng, quantum_reg, state=[j % 2])  # Try both states.

            # Cheat and make sure the results are all the same
            eng.flush()
            result1 = np.array(eng.backend.cheat()[1])
            stabilizer.apply_stabilizer_circuit(eng, quantum_reg, binary[i])
            result2 = np.array(eng.backend.cheat()[1])
            assert np.all(np.abs(result1 - result2) < 1e-3)

            register = stabilizer.decoding_circuit(eng, quantum_reg, add_ancilla_bits=True,
                                                  deallocate_nqubits=True)
            All(Measure) | register
            assert int(register[-1]) == j % 2


def test_applying_stabilizer_circuit_on_4_2_code():
    r"""Test applying stabilier circuit on 4 2 code by obtaining the wave function."""
    n, k = 4, 2
    binary = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                       [1, 1, 1, 1, 1, 0, 0, 1]])
    stabilizer = StabilizerCode(binary, n, k)
    for j in range(0, 20):
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(4)
        eng.flush()

        # Apply Encoding circuit on random state.
        random_state = [j % 2, (j + np.random.randint(1, 4)) % 2]
        stabilizer.encoding_circuit(eng, quantum_reg, state=random_state)

        # Get the wave-function before applying stabilizer
        result1 = np.array(eng.backend.cheat()[1])
        # Apply stabilizer which should affected the result.
        stabilizer.apply_stabilizer_circuit(eng, quantum_reg, binary[0])
        stabilizer.apply_stabilizer_circuit(eng, quantum_reg, binary[1])
        stabilizer.apply_stabilizer_circuit(eng, quantum_reg, stabilizer.normal_form[1])
        result2 = np.array(eng.backend.cheat()[1])
        assert np.all(np.abs(result2 - result1) < 1e-4)

        register = stabilizer.decoding_circuit(eng, quantum_reg, add_ancilla_bits=True)
        All(Measure) | register
        assert int(register[-1]) == random_state[-1]
        assert int(register[-2]) == random_state[-2]


def test_applying_stabilizer_circuit_on_random_code():
    code = ["ZYI", "ZIY"]
    binary = np.array([[0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1]])
    stabilizer = StabilizerCode(binary, 3, 1)

    for j in range(0, 20):
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(3)
        eng.flush()

        # Apply stabilizer.
        stabilizer.encoding_circuit(eng, quantum_reg, state=[0])
        eng.flush()

        result1 = np.array(eng.backend.cheat()[1])
        stabilizer.apply_stabilizer_circuit(eng, quantum_reg, stabilizer.normal_form[0])
        stabilizer.apply_stabilizer_circuit(eng, quantum_reg, stabilizer.normal_form[1])
        result2 = np.array(eng.backend.cheat()[1])

        register = stabilizer.decoding_circuit(eng, quantum_reg, add_ancilla_bits=True)
        All(Measure) | register
        assert int(register[-1]) == 0


def test_encoding_circuit_4_2_maps_to_plus_one_eigenspace():
    r"""Test the encoding cirucit on [4, 2] code that it maps to the right code-space."""
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)

    for state in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(4)
        eng.flush()
        code.encoding_circuit(eng, quantum_reg, state=state)
        eng.flush()
        wave_function = eng.backend.cheat()[1]
        All(Measure) | quantum_reg

        # Get stabilizer in matrix format.
        stabilizers = StabilizerCode.binary_rep_to_pauli_mat(code.normal_form)
        print([StabilizerCode.binary_rep_to_pauli_str(x) for x in code.normal_form])
        print([StabilizerCode.binary_rep_to_pauli_str(x) for x in binary_rep])
        for stab in stabilizers:
            print(wave_function - stab.dot(wave_function))
            assert np.all(np.abs(wave_function - stab.dot(wave_function)) < 1e-3)


def test_encoding_circuit_bit_flip_maps_to_plus_one_eigenspace():
    binary_rep = np.array([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]])
    stabilizer = StabilizerCode(binary_rep, 3, 1)

    for state in [[0], [1]]:
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(3)
        eng.flush()
        stabilizer.encoding_circuit(eng, quantum_reg, state=state)
        eng.flush()
        wave_function = eng.backend.cheat()[1]
        All(Measure) | quantum_reg

        # Get stabilizer in matrix format.
        stabilizers = StabilizerCode.binary_rep_to_pauli_mat(binary_rep)

        for stab in stabilizers:
            assert np.all(np.abs(wave_function - stab.dot(wave_function)) < 1e-3)


def test_encoding_circuit_shor_maps_to_plus_one_eigenspace():
    r"""Test that the encoding ciruict for shor is in the plus one eigenspace."""
    binary_rep = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                           [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    code = StabilizerCode(binary_rep, 9, 1)

    for state in [[0], [1]]:
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(9)
        eng.flush()
        code.encoding_circuit(eng, quantum_reg, state=state)
        eng.flush()
        wave_function = eng.backend.cheat()[1]
        All(Measure) | quantum_reg

        # Get stabilizer in matrix format.
        stabilizers = StabilizerCode.binary_rep_to_pauli_mat(code.normal_form)
        for stab in stabilizers:
            # Check that the wavefunction is in the plus one eigenspace of each stabilizer.
            assert np.all(np.abs(wave_function - stab.dot(wave_function)) < 1e-3)


def test_encoding_circuit_on_5_1_maps_to_plus_one_eigenspace():
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    n, k = 5, 1
    code = StabilizerCode(binary_rep, n, k)

    for state in [[0], [1]]:
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(n)
        eng.flush()
        code.encoding_circuit(eng, quantum_reg, state=state)
        eng.flush()
        wave_function = eng.backend.cheat()[1]
        All(Measure) | quantum_reg

        # Get stabilizer in matrix format.
        stabilizers = StabilizerCode.binary_rep_to_pauli_mat(code.binary_rep)
        for stab in stabilizers:
            print(wave_function + stab.dot(wave_function))
            # Check that the wavefunction is in the plus one eigenspace of each stabilizer.
            assert np.all(np.abs(wave_function - stab.dot(wave_function)) < 1e-3)


def test_encoding_circuit_5_1_is_the_same_from_book():
    r"""Test the encoding circuit on the [5, 1] code is the same as the book."""
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)

    def circuit_from_the_book(eng, reg):
        HGate() | reg[0]
        with Control(eng, reg[0]):
            QubitOperator('Y4', -1.j) | reg

        HGate() | reg[1]
        with Control(eng, reg[1]):
            XGate() | reg[4]

        HGate() | reg[2]
        with Control(eng, reg[2]):
            ZGate() | reg[0]
            ZGate() | reg[1]
            XGate() | reg[4]

        HGate() | reg[3]
        with Control(eng, reg[3]):
            ZGate() | reg[0]
            ZGate() | reg[2]
            QubitOperator('Y4', -1.j) | reg

    for state in [[0], [1]]:
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(n)
        eng.flush()
        code.encoding_circuit(eng, quantum_reg, state=state)
        eng.flush()

        wave_function_code = np.array(eng.backend.cheat()[1])
        All(Measure) | quantum_reg

        # From the book.
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(n)
        eng.flush()
        if state[0] == 1:
            XGate() | quantum_reg[-1]
        circuit_from_the_book(eng, quantum_reg)
        eng.flush()

        wave_function_book = np.array(eng.backend.cheat()[1])
        All(Measure) | quantum_reg
        assert np.all(np.abs(wave_function_book - wave_function_code) < 1e-4)


def circuit_from_the_book(eng, register):
    eng.flush()
    HGate() | register[0]
    with Control(eng, register[0]):
        QubitOperator('Y4', -1.j) | register
        QubitOperator('Y5', -1.j) | register
        XGate() | register[6]
        ZGate() | register[7]
    HGate() | register[1]
    with Control(eng, register[1]):
        QubitOperator('Y4', -1.j) | register
        XGate() | register[5]
        ZGate() | register[6]
        QubitOperator('Y7', -1.j) | register
    HGate() | register[2]
    with Control(eng, register[2]):
        ZGate() | register[1]
        QubitOperator('Y4', -1.j) | register
        QubitOperator('Y6', -1.j) | register
        XGate() | register[7]
    HGate() | register[3]
    with Control(eng, register[3]):
        ZGate() | register[2]
        ZGate() | register[4]
        QubitOperator('Y5', -1.j) | register
        XGate() | register[6]
        XGate() | register[7]


def test_encoding_circuit_8_3_maps_to_plus_one_eigenspace():
    n, k = 8, 3
    binary_rep = np.array([
                           [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],])
    code = StabilizerCode(binary_rep, n, k)

    for state in [[0, 0, 0]]:
        print("State is ", state)
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(n)
        eng.flush()
        # code.encoding_circuit(eng, quantum_reg, state=state)
        circuit_from_the_book(eng, quantum_reg)
        wave_function = eng.backend.cheat()[1]
        All(Measure) | quantum_reg

        # Get stabilizer in matrix format.
        stabilizers = StabilizerCode.binary_rep_to_pauli_mat(code.normal_form)
        for i, stab in enumerate(stabilizers):
            # Check that the wavefunction is in the plus one eigenspace of each stabilizer.
            assert np.all(np.abs(wave_function - stab.dot(wave_function)) < 1e-3)


def test_encoding_circuit_8_3_is_the_same_from_book():
    r"""Test the 8_3 circuit is the same as the book."""
    n, k = 8, 3
    binary_rep = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)

    for state in [[0, 0, 0]]:
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(n)
        eng.flush()
        code.encoding_circuit(eng, quantum_reg, state=state)
        wave_function_code = np.array(eng.backend.cheat()[1])
        All(Measure) | quantum_reg

        # From the book.
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(n)
        eng.flush()
        circuit_from_the_book(eng, quantum_reg)
        eng.flush()

        wave_function_book = np.array(eng.backend.cheat()[1])
        All(Measure) | quantum_reg
        assert np.all(np.abs(wave_function_book - wave_function_code) < 1e-4)


def test_syndrome_measurement_on_9_1_code():
    r"""Test syndrome measurement circuit of shor code."""
    n, k = 9, 1
    binary_rep = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                           [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    code = StabilizerCode(binary_rep, n, k)

    for state in [[0], [1]]:
        for _ in range(0, 50): # Try this fifty times.
            eng = MainEngine()
            quantum_reg = eng.allocate_qureg(n)
            eng.flush()

            code.encoding_circuit(eng, quantum_reg, state=state)
            wave_function_before = np.array(eng.backend.cheat()[1])

            # Apply syndrome measurement based on normal form stabilizers.
            random_int = np.random.randint(0, 8)
            measurement = code.single_syndrome_measurement(eng, quantum_reg,
                                                           code.normal_form[random_int])
            wave_function_after = np.array(eng.backend.cheat()[1])
            assert np.all(np.abs(wave_function_before - wave_function_after) < 1e-4)
            assert measurement == 0
            All(Measure) | quantum_reg

            # Apply Error .
            eng = MainEngine()
            quantum_reg = eng.allocate_qureg(n)
            eng.flush()

            code.encoding_circuit(eng, quantum_reg, state=state)  # Encode
            XGate() | quantum_reg[1]  # Apply Error.
            # Apply syndrome measurement based on normal form stabilizers.
            found_one = False
            for i in range(0, 8):
                measurement = code.single_syndrome_measurement(eng, quantum_reg,
                                                               code.normal_form[i])
                if measurement == 1:
                    found_one = True
            assert found_one
            All(Measure) | quantum_reg

            # Apply logical operator.
            eng = MainEngine()
            quantum_reg = eng.allocate_qureg(n)
            eng.flush()

            code.encoding_circuit(eng, quantum_reg, state=state)  # Encode
            code.logical_circuit(eng, quantum_reg, "X0")
            measurement = code.single_syndrome_measurement(eng, quantum_reg,
                                                           code.normal_form[random_int])
            assert measurement == 0
            All(Measure) | quantum_reg


def test_syndrome_measurement_on_8_3_code():
    r"""Test syndrome measurement circuit of [8, 3] code."""
    n, k = 8, 3
    binary_rep = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)

    for state in [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]:
        for _ in range(0, 50): # Try this fifty times.
            eng = MainEngine()
            quantum_reg = eng.allocate_qureg(n)
            eng.flush()

            code.encoding_circuit(eng, quantum_reg, state=state)
            wave_function_before = np.array(eng.backend.cheat()[1])

            # Apply syndrome measurement based on normal form stabilizers.
            random_int = np.random.randint(0, 5)
            measurement = code.single_syndrome_measurement(eng, quantum_reg,
                                                           code.normal_form[random_int])
            wave_function_after = np.array(eng.backend.cheat()[1])
            assert np.all(np.abs(wave_function_before - wave_function_after) < 1e-4)
            assert measurement == 0
            All(Measure) | quantum_reg

            # Apply logical operator.
            eng = MainEngine()
            quantum_reg = eng.allocate_qureg(n)
            eng.flush()

            code.encoding_circuit(eng, quantum_reg, state=state)  # Encode
            random_logical_op = np.random.choice(["X0", "X1", "X2", "Z0", "Z1", "Z2", "Y0", "Y1",
                                                  "Y2"])
            code.logical_circuit(eng, quantum_reg, random_logical_op)
            measurement = code.single_syndrome_measurement(eng, quantum_reg,
                                                           code.normal_form[random_int])
            assert measurement == 0
            All(Measure) | quantum_reg


def test_logical_circuit_on_bit_flip_code():
    r"""Test logical circuit on the bit flip code."""
    n, k = 3, 1
    binary_rep = np.array([[0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 1, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)

    # Test apply "X0" To encoded |0> Is equivalent to encoded |1>
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [0])
    wavefunc_zero = np.array(eng.backend.cheat()[1])
    code.logical_circuit(eng, quantum_reg, "X0")
    wave_function_actual = np.array(eng.backend.cheat()[1])
    All(Measure) | quantum_reg

    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [1])
    wave_function_desired = np.array(eng.backend.cheat()[1])

    assert np.all(np.abs(wave_function_actual - wave_function_desired) < 1e-4)
    All(Measure) | quantum_reg

    # Test apply "X0" To encoded |1> Is equivalent to encoded |0>.
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [1])
    code.logical_circuit(eng, quantum_reg, "X0")
    wave_function_actual = np.array(eng.backend.cheat()[1])
    assert np.all(np.abs(wave_function_actual - wavefunc_zero) < 1e-4)
    All(Measure) | quantum_reg


def test_logical_circuit_on_5_1_code():
    r"""Test logical circuit on the 5, 1 code."""
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)

    # Test apply "X0" To encoded |0> Is equivalent to encoded |1>
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [0])
    wavefunc_zero = np.array(eng.backend.cheat()[1])
    code.logical_circuit(eng, quantum_reg, "X0")
    wave_function_actual = np.array(eng.backend.cheat()[1])
    All(Measure) | quantum_reg

    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [1])
    wave_function_desired = np.array(eng.backend.cheat()[1])

    assert np.all(np.abs(wave_function_actual - wave_function_desired) < 1e-4)
    All(Measure) | quantum_reg

    # Test apply "X0" To encoded |1> Is equivalent to encoded |0>.
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [1])
    code.logical_circuit(eng, quantum_reg, "X0")
    wave_function_actual = np.array(eng.backend.cheat()[1])
    assert np.all(np.abs(wave_function_actual - wavefunc_zero) < 1e-4)
    All(Measure) | quantum_reg

    # Test apply "Z0" to encoded |1> is equivalent to encoded -|1>
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [1])
    wave_func_before = np.array(eng.backend.cheat()[1])
    code.logical_circuit(eng, quantum_reg, "Z0")
    wave_function_actual = np.array(eng.backend.cheat()[1])
    assert np.all(np.abs(wave_function_actual + wave_func_before) < 1e-4)
    All(Measure) | quantum_reg

    # Test apply "Z0" to encoded |0> is equivalent to encoded |0>
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [0])
    wave_func_before = np.array(eng.backend.cheat()[1])
    code.logical_circuit(eng, quantum_reg, "Z0")
    wave_function_actual = np.array(eng.backend.cheat()[1])
    assert np.all(np.abs(wave_function_actual - wave_func_before) < 1e-4)
    All(Measure) | quantum_reg


def test_logical_circuit_on_4_2_code():
    r"""Test logical circuit on the 4, 2 code."""
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k)

    # Test apply "X1" To encoded |00> Is equivalent to encoded |01>
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [0, 0])
    wavefunc_zero = np.array(eng.backend.cheat()[1])
    code.logical_circuit(eng, quantum_reg, "X1")
    wave_function_actual = np.array(eng.backend.cheat()[1])
    All(Measure) | quantum_reg

    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [0, 1])
    wave_function_desired = np.array(eng.backend.cheat()[1])
    assert np.all(np.abs(wave_function_actual - wave_function_desired) < 1e-4)
    All(Measure) | quantum_reg

    # Test apply "X0" To encoded |10> Is equivalent to encoded |00>.
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [1, 0])
    code.logical_circuit(eng, quantum_reg, "X0")
    wave_function_actual = np.array(eng.backend.cheat()[1])
    assert np.all(np.abs(wave_function_actual - wavefunc_zero) < 1e-4)
    All(Measure) | quantum_reg

    # Test apply "Z1" to encoded |11> is equivalent to encoded -|11>
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [1, 1])
    wave_func_before = np.array(eng.backend.cheat()[1])
    code.logical_circuit(eng, quantum_reg, "Z1")
    wave_function_actual = np.array(eng.backend.cheat()[1])
    assert np.all(np.abs(wave_function_actual + wave_func_before) < 1e-4)
    All(Measure) | quantum_reg

    # Test apply "Z0" to encoded |01> is equivalent to encoded |01>
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(n)
    eng.flush()
    code.encoding_circuit(eng, quantum_reg, [0, 1])
    wave_func_before = np.array(eng.backend.cheat()[1])
    code.logical_circuit(eng, quantum_reg, "Z0")
    wave_function_actual = np.array(eng.backend.cheat()[1])
    assert np.all(np.abs(wave_function_actual - wave_func_before) < 1e-4)
    All(Measure) | quantum_reg


if __name__ == "__main__":
    # Test that use the encoder but somehow work.
    # test_decoding_circuit_with_encoding_circuit_perserves_state_8_3_code()
    # test_decoding_circuit_with_encoding_circuit_perserves_state_shor_code()
    # test_decoding_circuit_with_encoding_circuit_perserves_state_4_2_code()
    # test_decoding_circuit_with_encoding_circuit_perserves_state_5_1_code()
    # test_syndrome_measurement_on_9_1_code()
    # test_syndrome_measurement_circuit_on_4_2_code()
    # test_syndrome_measurement_circuit_on_bit_flip_code()
    # test_syndrome_measurement_circuit_on_phase_flip_code()
    # test_syndrome_measurement_on_8_3_code()
    # test_logical_circuit_on_4_2_code()
    # test_logical_circuit_on_5_1_code()
    # test_logical_circuit_on_bit_flip_code()


    # test_kraus_operators_for_encoder()
    # test_applying_stabilizer_circuit_on_4_2_code()
    # test_syndrome_measurement_on_9_1_code()
    # test_syndrome_measurement_circuit_on_4_2_code()
    # assert 1 == 0
    # test_applying_stabilizer_circuit_on_random_code()
    #
    test_encoding_circuit_bit_flip_maps_to_plus_one_eigenspace()
    test_encoding_circuit_4_2_maps_to_plus_one_eigenspace()
    test_encoding_circuit_on_5_1_maps_to_plus_one_eigenspace()
    test_encoding_circuit_shor_maps_to_plus_one_eigenspace()
    test_encoding_circuit_8_3_is_the_same_from_book()
    test_encoding_circuit_5_1_is_the_same_from_book()
    # test_encoding_circuit_8_3_maps_to_plus_one_eigenspace()

    # test_applying_stabilizer_circuit_on_random_code()
    # test_applying_stabilizer_circuit_on_4_2_code()
    # test_applying_circuit_stabilizers_on_bitflip_code()
    # test_decoding_circuit_with_encoding_circuit_perserves_state_4_2_code()
    # test_syndrome_measurement_circuit_on_4_2_code()
    # test_syndrome_measurement_circuit_on_phase_flip_code()
    # test_syndrome_measurement_circuit_on_bit_flip_code()
    pass
