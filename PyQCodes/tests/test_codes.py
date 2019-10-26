import numpy as np
import qecc as q

from PyQCodes.codes import *



def test_binary_representation():
    """Test binary representation of stabilizer codes."""
    code = ["ZZI", "ZIZ"]
    binary_rep = np.array([[0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 1, 0, 1]])


    code = ["XIXI", "ZIXX"]
    binary_rep = np.array([[1, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0, 0]])

    code = ["YII"]
    binary_rep = np.array([[1, 0, 0, 1, 0, 0]])


def test_two_pauli_operators_commute_with_inner_product():
    """Test two pauli operators commute or not."""
    # Anti-Commuting operators
    two_pauli_operators = ["XXY", "ZII"]
    binary_rep_xxy = np.array([1, 1, 1, 0, 0, 1], dtype=np.int)
    binary_rep_zii = np.array([0, 0, 0, 1, 0, 0], dtype=np.int)
    desired_answer = binary_rep_xxy.dot(binary_rep_zii) % 2 # Should be One

    # Commuting Operators
    two_pauli_operators = ["XIIX", "ZIIZ"]
    binary_rep_xiix = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int)
    binary_rep_ziiz = np.array([0, 0, 0, 0, 1, 0, 0, 1], dtype=np.int)
    desired_answer = binary_rep_xiix.dot(binary_rep_ziiz) % 2 # Should be zero.


def test_stabilizer_code_is_commutative():
    r"""Test stabilizer code is commutative set"""
    stabilizers = ["ZZI", "ZIZ"]
    binary_rep = np.array([[0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 1, 0, 1]], dtype=np.int)

    desired_answer = binary_rep[0,:3].dot(binary_rep[1, 3:])
    desired_answer += binary_rep[0, 3:].dot(binary_rep[1, :3])
    desired_answer = desired_answer % 2 # Should be zero


    stabilizers = ["YZI", "XII"]
    binary_rep = np.array([[1, 0, 0, 1, 1 ,0],
                           [1, 0, 0, 0, 0, 0]], dtype=np.int)
    desired_answer = binary_rep[0,:3].dot(binary_rep[1, 3:])
    desired_answer += binary_rep[0, 3:].dot(binary_rep[1, :3])
    desired_answer = desired_answer % 2 # Should be one


def test_generator_set_pauli_elements():
    r"""Test that the generators are obtained."""
    stabilizers = ["ZZ", "ZI"]
    binary_rep = np.array([[0, 0, 1, 1],
                           [0, 0, 1, 0]], dtype=np.int)
    stabilizer = StabilizerCode(binary_rep, 2, 1, None, None)
    actual = stabilizer.generator_set_pauli_elements()

    desired = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert np.all(np.abs(actual - np.array(desired)) < 1e-5)

    stabilizers = ["ZZI", "ZIZ"]
    binary_rep = np.array([[0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 1, 0, 1]], dtype=np.int)
    stabilizer = StabilizerCode(binary_rep, 3, 1, None, None)
    desired = np.eye(6, dtype=np.int)
    assert np.all(np.abs(stabilizer.generator_set_pauli_elements() - desired) < 1e-5)


def test_code_concatenation():
    pass


def test_normalizer():
    r"""Test getting the normalizer of the stabilizer group."""
    code = ["ZIZ", "ZZI"]
    binary_rep = np.array([[0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 1, 0, 1]], dtype=np.int)
    normalizer_elements = ["XXX", 'ZZZ']
    # Order it so it matches the algorithm.
    binary_rep_norm = np.array([[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    stabilizer = StabilizerCode(binary_rep, 3, 1, None, None)
    actual = stabilizer.normalizer()
    # assert np.all(np.abs(actual - binary_rep_norm) < 1e-5)
    print(actual)
    stab = q.StabilizerCode(['ZZI', 'IZZ'], ['XXX'], ['ZZZ'])
    print([x for x in stab.normalizer_group()])


def test_kraus_operators_for_encoder():
    r"""Test the kraus operators for encoder."""
    # Test bit-flip code.
    code = ["ZZI", "ZIZ"]
    binary_rep = np.array([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]])
    stabilizr = StabilizerCode(binary_rep, 3, 1, None, None)
    desired = np.array([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]])
    actual = stabilizr.encode_krauss_operators()
    assert np.all(np.abs(desired - actual) < 1e-5)
    # Test that it is an isometry
    assert np.all(np.abs(actual.T.conj().dot(actual) - np.eye(2)) < 1e-5)

    # Test Phase-Flip Code.
    binary_rep = np.array([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]])
    stabilizr = StabilizerCode(binary_rep, 3, 1, None, None)
    actual = stabilizr.encode_krauss_operators()

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
    stabilizr = StabilizerCode(binary_rep, 9, 1, None, None)
    actual = stabilizr.encode_krauss_operators()
    state1 = np.array([1., 0, 0, 0, 0, 0, 0, 1])  / np.sqrt(2)
    state2 = np.array([1., 0, 0, 0, 0, 0, 0, -1]) / np.sqrt(2)
    basis_vec1 = np.kron(state1, np.kron(state1, state1))
    basis_vec2 = np.kron(state2, np.kron(state2, state2))
    coeff1 = np.linalg.lstsq(actual, basis_vec1, rcond=None)[0]
    coeff2 = np.linalg.lstsq(actual, basis_vec2, rcond=None)[0]
    assert np.all(np.abs(coeff1) > 1e-5)
    assert np.all(np.abs(coeff2) > 1e-5)

    # Test Cat Code n = 2
    n, k = 2, 1
    binary_rep = np.array([[0, 0, 1, 1]])
    stabilizr = StabilizerCode(binary_rep, n, k, None, None)
    actual = stabilizr.encode_krauss_operators()
    desired = np.array([[1., 0.], [0., 0.], [0., 0.], [0., 1.]])
    assert np.all(np.abs(actual - desired) < 1e-4)


def test_encoding_circuit():
    code = ["ZZI", "ZIZ"]
    desired_answer = None #CNOT_{0, 2} CNOTE{0, 1}


def test_decoding_circuit():
    # Opposite of encoding circuit
    pass


def test_measurement_circuit():
    from projectq.ops import H, Measure, All, X
    from projectq import MainEngine
    eng = MainEngine()
    quantum_reg = eng.allocate_qureg(3)

    # Test bit-flip code.

    code = ["ZZI", "ZIZ"]
    binary_rep = np.array([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]])
    stabilizr = StabilizerCode(binary_rep, 3, 1, None, None)

    # Test first eigenvector basis.
    measure_result = stabilizr.syndrome_measurement(eng, quantum_reg, binary_rep[0])
    measure_result2 = stabilizr.syndrome_measurement(eng, quantum_reg, binary_rep[1])
    assert measure_result == 0
    assert measure_result2 == 0

    # Test eigenvector basis.
    for _ in range(0, 1000):
        eng = MainEngine()
        quantum_reg = eng.allocate_qureg(3)
        X | quantum_reg[0]
        X | quantum_reg[2]
        measure_result = stabilizr.syndrome_measurement(eng, quantum_reg, binary_rep[1])
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
    code = StabilizerCode(binary_rep, n, k, None, None)
    desired = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                        [0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 1, 1, 1]])
    output = code._gaussian_elimination_first_block()
    assert np.all(np.abs(desired - output) < 1e-5)

    # Test on [4, 2, 2] Code "XZZX", "YXXY"
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    desired = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                        [0, 1, 1, 0, 1, 1, 1, 1]])
    code = StabilizerCode(binary_rep, n, k, None, None)
    output = code._gaussian_elimination_first_block()
    assert np.all(np.abs(desired - output) < 1e-5)

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
    code = StabilizerCode(binary_rep, n, k, None, None)
    output = code._gaussian_elimination_first_block()
    assert np.all(np.abs(desired - output) < 1e-5)

    # # Test on Steane Code from Nielsen and Chaung.
    # n, k = 7, 1
    # binary_rep = np.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #                        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    #                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
    #                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]])
    # desired = np.array([[1, 0, 0, 0, 1, 1, 1,   0, 0, 0, 0, 0, 0, 0],
    #                     [0, 1, 0, 1, 0, 1, 1,   0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 1, 1, 1, 1, 0,   0, 0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0, 0,   1, 0, 1, 1, 0, 0, 1],
    #                     [0, 0, 0, 0, 0, 0, 0,   0, 1, 1, 0, 1, 0, 1],
    #                     [0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 0, 0, 1, 0]])
    # code = StabilizerCode(binary_rep, n, k, None, None)
    # output = code._gaussian_elimination_first_block()
    # print(output)
    # print(output - desired)
    # assert np.all(np.abs(desired - output) < 1e-5)


def test_gaussian_elimination_second_block_of_binary_representation():
    r"""Test gaussian elimination of the second block of binary representation."""
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k, None, None)
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
    code = StabilizerCode(binary_rep, n, k, None, None)
    output = code._gaussian_elimination_second_block(gaus_eliminated, 3)
    assert np.all(np.abs(output - gaus_eliminated) < 1e-5)


def test_standard_normal_form():
    r"""Test putting the binary representation into standard normal form."""
    n, k = 7, 1
    binary_rep = np.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]])

    # Test already put in standard normal form.
    # desired = np.array([
    #             [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
    #             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]])
    # code = StabilizerCode(binary_rep, n, k, None, None)
    # output = code._standard_normal_form()
    # print(output)
    # print(output - desired)
    # assert np.all(np.abs(output  -desired) < 1e-5)

    # Test on [4, 2, 2] Code "XZZX", "YXXY"
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    desired = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                        [0, 1, 1, 0, 1, 1, 1, 1]])
    code = StabilizerCode(binary_rep, n, k, None, None)
    output, rank = code._standard_normal_form()
    assert np.all(np.abs(desired - output) < 1e-5)
    assert rank == 2

    # Five qubit code: "XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"
    n, k = 5, 1
    binary_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k, None, None)
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
    code = StabilizerCode(binary_rep, n, k, None, None)
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
    code = StabilizerCode(binary_rep, n, k, None, None)
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
    code = StabilizerCode(binary_rep, n, k, None, None)
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
    code = StabilizerCode(binary_rep, n, k, None, None)

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
    code = StabilizerCode(binary_rep, n, k, None, None)
    actual_X, actual_Z = code.logical_operators()
    desired_X = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 1, 0]])
    desired_Z = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    assert np.all(np.abs(desired_X - actual_X) < 1e-5)
    assert np.all(np.abs(desired_Z - actual_Z) < 1e-5)

    # Test [4, 2] Code from Gaitan book pg (131)
    n, k = 4, 2
    binary_rep = np.array([[1, 0, 0, 1, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0, 1]])
    code = StabilizerCode(binary_rep, n, k, None, None)
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
    code = StabilizerCode(binary_rep, n, k, None, None)
    actual_X, actual_Z = code.logical_operators()
    desired_X = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]])
    desired_Z = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]])
    assert np.all(np.abs(desired_X - actual_X) < 1e-5)
    assert np.all(np.abs(desired_Z - actual_Z) < 1e-5)


if __name__ == "__main__":
    test_logical_operators()
    test_matrix_blocks_standard_form()
    test_standard_normal_form()
    test_gaussian_elimination_second_block_of_binary_representation()
    # test_measurement_circuit()
    # test_normalizer()
    # test_generator_set_pauli_elements()
    # test_binary_rep_to_pauli_mat()
    # test_kraus_operators_for_encoder()