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
from abc import ABC, abstractmethod, abstractstaticmethod
from projectq.ops import CNOT, H, HGate, S, Measure, All, QubitOperator, C, XGate, YGate, ZGate
from projectq.meta import Control

import numpy as np
from scipy.sparse import kron, identity, csr_matrix
import warnings
from scipy.sparse.linalg import eigsh
import itertools


__all__ = ["QCode", "StabilizerCode"]


# Constants: Pauli Operators Plus Sparse Pauli Operators used throughout this file.
X, Y, Z, I = [np.array([[0., 1.], [1., 0.]]),
              np.array([[0., complex(0, -1.)], [complex(0., 1.), 0.]]),
              np.array([[1., 0.], [0, -1.]]), np.eye(2)]

XS, YS, ZS, IS = [csr_matrix(X, dtype=np.complex128),
                  csr_matrix(Y, dtype=np.complex128),
                  csr_matrix(Z, dtype=np.complex128),
                  identity(2, dtype=np.complex128, format="csr")]


class QCode(ABC):
    r"""Abstract base class for quantum codes."""

    def __init__(self, encoder, decoder, syndrome_measure, correct, n, k):
        r"""Contruct quantum codes based on basic properties."""
        pass

    @abstractmethod
    def encode(self, eng, register):
        r"""Represent the encoding process from k-qubits to n-qubits."""
        pass

    @abstractmethod
    def decode(self, eng, register):
        r"""Represent the decoding process from n-qubits to k-qubits."""
        pass

    @abstractmethod
    def syndrome_measure(self, eng, register, measurement):
        r"""Represent the syndrome measurement of the measurement."""
        pass

    @abstractmethod
    def correct(self, eng, register, operator):
        r"""Correct the error by applying operator."""
        pass


class StabilizerCode():
    r"""
    Class for analyzing stabilizer codes.

    Attributes
    ----------
    n : int
        The number of encoded qubits from (n, k) code.
    k : int
        The number of un-encoded qubits from (n, k) code.
    stab_bin_rep : np.ndarray
        Two-dimensional binary array representing the binary representation of a stabilizer code.
    logical_x : np.ndarray
        Two-dimensional binary array representing the logical X operators on the encoded space.
    logical_z : np.ndarray
        Two-dimensional binary array representing the logical Z operators on the encoded space.

    Methods
    -------
    encode_krauss_operators :
        The encoding (isometry) matrix in standard-basis from 2^k-dim to 2^n-dim.
    kraus_operators_correcting_errors :
        Given a set of pauli-errors, construct the kraus operators representing the recovery
        operator.
    logical_operators :
        The binary representation of the logical operators
    encoding_circuit :
        Apply encoding circuit on ProjectQ engine from computational 2^k basis to the
        corresponding basis in the code space.
    single_syndrome_measurement :
        Perform a measurement on a specific stabilizer generator on the ProjectQ engine.
    decoding_circuit :
        Apply the decoding circuit on ProjectQ engine from the basis in the code-space to the
        standard computational basis.

    Examples
    --------
    # Example constructing the [1, 3] bit-flip code with stabilizers "ZZI", "ZIZ".
    >> binary_rep = np.array([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]])
    >> stabilizer = StabilizerCode(binary_rep, n=3, k=1)

    # This is the same as
    >> stab_generators = ["ZZI", "ZIZ"]
    >> stabilizer = StabilizerCode(binary_rep, n=3, k=1)

    """

    def __init__(self, stabilizer_group, n, k, logical_ops=None):
        r"""
        Construct the stabilizer code.

        Parameters
        ----------
        stabilizer_group : array or list
            Binary Representation of stabilizer groups or list of pauli strings.

        n : int
            Number of encoded qubits.

        k : int
            Number of un-encoded qubits.

        logical_ops : list, optional
            List of two arrays [LX, LZ], where LX is the binary representation of the logical x
            operators and LZ is the binary representation of the logical z operators. If None,
            then the logical operators will be found using the standard normal form.

        Examples
        --------
        Construct the phase-flip code.
        >> stab = ["XXI", "XIX"]
        >> code = StabilizerCode(stab, 3, 1)

        Alternatively, can provide the binary representation of the stabilizers "XXI" and "XIX".
        >> binary_rep = [[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]]
        >> code = StabilizerCode(binary_rep, 3, 1)

        If one wants to add their own logical X and logical Z operators.
        >> logical_x = np.array([[1, 0, 0, 0, 0, 0]])
        >> logical_z = np.array([[0, 0, 0, 1, 1, 1]])
        >> code = StabilizerCode(stab, 3, 1, [logical_x, logical_z])

        Raises
        ------
        AssertionError :
            Raises an assertion error if the stabilizer elements do not commute with one another or
            Stabilizer do not match the dimension of the code parameters.

        """
        if isinstance(stabilizer_group[0], str):
            # Convert from string to binary representations.
            stabilizer_group = StabilizerCode.pauli_str_to_binary_representation(stabilizer_group)
        elif isinstance(stabilizer_group[0], (list, np.ndarray)):
            stabilizer_group = np.array(stabilizer_group)
            assert stabilizer_group.shape[1] == 2*n, "Stabilizer should match parameter n."
            assert stabilizer_group.shape[0] == n - k, "Number of stabilizers should be n - k."

        self._stab_bin_rep = stabilizer_group
        self.numb_stab = self.stab_bin_rep.shape[0]
        self.dim_code_space = n - self.numb_stab
        self._n = n
        self._k = k
        assert self._is_stabilizer_code()
        self._pauli_stab = [StabilizerCode.binary_rep_to_pauli_str(x) for x in stabilizer_group]

        # The attributes from constructing the standard normal form, its rank and logical operators.
        self.normal_form, self.rank = self._standard_normal_form()
        if logical_ops is None:
            self._logical_x, self._logical_z = self.logical_operators()
        else:
            assert isinstance(logical_ops[0], np.ndarray), "X Logical operators should be in a " \
                                                           "numpy array."
            assert isinstance(logical_ops[1], np.ndarray), "Z logical operators should be in a " \
                                                           "numpy array."
            assert logical_ops[0].shape[1] == 2*n, "Number of columns of X logical operators " \
                                                   "should be 2*n."
            assert logical_ops[1].shape[1] == 2*n, "Number of columns of Z logical operators " \
                                                   "should be 2*n."
            self._logical_x, self._logical_z = logical_ops

    @property
    def n(self):
        r"""Obtain the total number of encoded qubits."""
        return self._n

    @property
    def k(self):
        r"""Obtain the number of unencoded qubits."""
        return self._k

    @property
    def stab_bin_rep(self):
        r"""Obtain the original binary representation provided."""
        return self._stab_bin_rep

    @property
    def pauli_stab(self):
        r"""Return the stabilizer in pauli string format."""
        return self._pauli_stab

    @property
    def logical_x(self):
        r"""Obtain the logical X operators."""
        return self._logical_x

    @property
    def logical_z(self):
        r"""Obtain the logical Z operators."""
        return self._logical_z

    def __iter__(self):
        r"""Iterate through the stabilizer generators in the normal form."""
        for stab in self.normal_form:
            yield stab

    def _is_stabilizer_code(self):
        r"""
        Check if all pauli elements in the stabilizer group are commutative with one another.

        Return
        ------
        bool
            True if pauli elements are commutative inside the stabilizer group.

        """
        is_stabilizer_code = True
        for i in range(0, self.numb_stab):
            for j in range(0, self.numb_stab):
                if i != j:  # Don't compare the same stabilizer elements to itself.
                    paul1 = self.stab_bin_rep[i]
                    paul2 = self.stab_bin_rep[j]
                    # Value of zero (False) indicates they they commute.
                    does_commute = StabilizerCode.inner_prod(paul1, paul2)
                    if does_commute:
                        is_stabilizer_code = False
                        break
        return is_stabilizer_code

    def encode_krauss_operators(self, sparse=False):
        r"""
        Isometry/Kraus operator for the encoder.

        Parameters
        ----------
        sparse : bool
            True, return the encoder map as a sparse matrix.

        Returns
        -------
        np.array :
            Returns a numpy array representing the encoder map from k-qubits to n-qubits,
            given a (n, k)-parameter code.

        Notes
        -----
        - The isometry operator is one whose columns span the code-space and are orthonormal to
        one another. The code-space is the sub-space of vectors that are in the plus one
        eigenspace of each stabilizer generator. The orthonormal basis is found by simultaneously
        diagonalizing each stabilizer generator S_1, ..., S_{n-k} together.

        """
        def _obtain_eigenspace_one(eigen, vecs):
            return vecs[:, np.abs(eigen - 1,) < 1e-5]

        # Diagonalize first stabilizer element and get only plus one eigenspace.
        stabilizers = StabilizerCode.binary_rep_to_pauli_mat(self.stab_bin_rep)
        stab1 = stabilizers[0]
        eigs, evecs = np.linalg.eigh(stab1)
        evecs = _obtain_eigenspace_one(eigs, evecs)

        # Continue Simutaneously Diagonalizing
        for i in range(1, self.numb_stab):
            conju = evecs.conjugate().T.dot(stabilizers[i].dot(evecs))
            e1, ev1 = np.linalg.eigh(conju)  # Get the eigenvectors ev1 and eigenvalues e1
            ev1 = _obtain_eigenspace_one(e1, ev1)  # Get eigenvectors that are plus-one eigenvalues.
            evecs = evecs.dot(ev1)

        if sparse:
            return csr_matrix(evecs, shape=evecs.shape, dtype=np.complex128)
        return evecs

    def kraus_operators_correcting_errors(self, pauli_errors, sparse):
        r"""
        Obtain The Kraus operators that can correct the errors provided, based on the stabilizers.

        Given a set of pauli errors, it will find which stabilizer element that commutes with it.
        If it commutes, then it returns the stabilizer multiplied by the error (a new
        kraus operator). If it doesn't commute, then just the pauli error is placed in the list.

        Parameters
        ----------
        pauli_errors : list of str
            List of strings representing pauli errors or the kraus operators that are scalar
            multiples of a pauli matrix. Can be single qubit or higher, as long as it matches the
            length of the stabilizers.
        sparse : bool
            True if the pauli errors provided are sparse.

        Returns
        -------
        kraus_operators : list
            Returns a list of kraus operators whose elements are either,
                the identity channel composed with a un-correctable error or
                the stabilizer composed (matrix multiplied) to a correctable error.

        Notes
        -----
        - This is primarily used for the effective channel method.

        """
        # Find a list of stabilizer that anti-commutes with pauli_errors
        kraus = []
        stabilizers = StabilizerCode.binary_rep_to_pauli_mat(self.stab_bin_rep, sparse=sparse)

        for i, error in enumerate(pauli_errors):
            found_decoder = False
            # Go through each stabilizer element.
            for p in stabilizers:
                # Check whether they commute or not.
                p_error = p.dot(error)
                error_p = error.dot(p)

                # Check if they are anti-commuting.
                if np.all(abs(error_p + p_error) < 1e-5):
                    # Append stabilier multipled by error for entire error-correcting procedure.
                    kraus.append(p.dot(error))
                    found_decoder = True  # Error is correctable.
                    break

            # If the error commutes with all stabilizer elements.
            if not found_decoder:
                kraus.append(error)
        return kraus

    def single_syndrome_measurement(self, eng, register, stabilizer):
        r"""
        Get the syndrome measurement of stabilizer element.

        Note that if the length of register is n, then this allocates a new qubit, does the
        circuit, then deletes the allocated qubit. If length of register is n+1, then it uses the
        last qubit in the register as a measurement ancilla.

        Stabilizer element is recommended to be in standard normal form.

        Parameters
        ----------
        eng : ProjectQ Engine
            The quantum circuit engine.
        register : ProjectQ Qureg or list of Qubit
            Either a quantum register or a list of qubits.
        stabilizer : (np.ndarray(2n,) or string)
            Binary representation of the stabilizer element or pauli string.

        Returns
        -------
        int :
            The measurement corresponding to the stabilizer element. Zero means it commutes and
            negative one means it anti-commutes.

        References
        ----------
        - See Quantum Error Correction Book By Daniel Lidar Page 72.

        """
        numb_qubits = len(eng.active_qubits)

        # The additional qubit is for measurement purposes.
        if numb_qubits != self.n and numb_qubits != self.n + 1:
            raise TypeError("Number of qubits allocated should match the number of encoded qubits n"
                            " from (n,k) code or match n + 1, where last qubit is used as an "
                            "ancilla.")

        # Allocate a new qubit for measurement, if it doesn't have it already.
        if numb_qubits == self.n:
            measure_qubit = eng.allocate_qubit()
        else:
            measure_qubit = register[-1]

        # Convert stabilizer element to pauli matrix and construct the circuit and measure it.
        pauli_str = stabilizer
        if isinstance(stabilizer, np.ndarray):
            pauli_str = self.binary_rep_to_pauli_str(np.array(stabilizer))
        print(pauli_str)
        H | measure_qubit
        eng.flush()
        with Control(eng, measure_qubit):
            for i, pauli_element in enumerate(pauli_str):
                if pauli_element == 'X':
                    XGate() | register[i]
                elif pauli_element == 'Y':
                    # ZGate() | register[i]
                    # XGate() | register[i]
                    QubitOperator('Y' + str(i), 1.) | register
                elif pauli_element == 'Z':
                    ZGate() | register[i]
                elif pauli_element == "I":
                    pass
                else:
                    raise RuntimeError("Pauli strings contained an unknown character.")
            eng.flush()
        H | measure_qubit
        eng.flush()
        Measure | measure_qubit
        eng.flush()
        result = int(measure_qubit)
        if numb_qubits == self.n:
            del measure_qubit
        return result

    def encoding_circuit(self, eng, register, state):
        r"""
        Apply the encoding circuit to map the k-qubit "state" to its n-qubit state.

        Parameters
        ----------
        state : list
            list of k-items, where each item is either 0 to 1 corresponding to the quantum state
            |x_1, ... , x_k>, where x_i is either zero or one.

        Notes
        -----
        - To construct the most optimal encoding circuit. The standard form for the stabilizer
                code needs to be constructed alongside the logical X operators.

        References
        ----------
        - See Gaitan book "Quantum Error-Correction and Fault Tolerant Quantum Computing."

        """
        assert len(state) == self.k, "State should be the number of unencoded qubits k."
        assert len(register) == self.n, "Number of qubits should be number of encoded qubits n."

        logical_x = self.logical_x
        # Construct The last k qubits to become the specified attribute 'state'.
        for i, binary in enumerate(state):
            assert binary in [0, 1], "state should be all binary elements."
            if binary == 1:
                XGate() | register[self.n - self.k + i]

        # Construct Controlled Unitary operators To Model Logical X Operators, this is only needed
        # when the rank is less than n- k.
        if self.rank < self.numb_stab:
            # TODO: Check this.
            print("the rank is", self.rank)
            print("Numb of stabs is ", self.numb_stab)

            for i in range(0, self.k):  # Go Thorough each un-coded.
                # Get the Controlled unitary operator
                controlled_op = logical_x[i, self.rank:self.n - self.k]
                with Control(eng, register[self.n - self.k + i]):
                    for j, binary in enumerate(controlled_op):
                        if binary == 1:
                            XGate() | register[self.rank + j]

        # Construct The application of stabilizer generators.
        # Go Through the first rank qubits, should be all initialized to zero, or go through
        # the first type 1 stabilizer generators.
        for i in range(0, self.rank):
            # Apply hadamard gate to every encoded qubit.
            HGate() | register[i]

            # Get pauli operator of normal stabilizer generator.
            pauli = self.binary_rep_to_pauli_str(self.normal_form[i])

            # Apply controlled operators with the ith-qubit being controlled.
            with Control(eng, register[i]):
                for j, pauli_op in enumerate(pauli):
                    if j != i:  # The ith qubit is controlled.
                        if pauli_op == 'X':
                            XGate() | register[j]
                        elif pauli_op == 'Y':
                            # ZGate() | register[j]
                            # XGate() | register[j]
                            # YGate() | register[j]
                            QubitOperator('Y' + str(j), 1.j) | register
                        elif pauli_op == 'Z':
                            #  Z Gate Acts trivially on |0000 \delta>
                            # if j < i:
                            ZGate() | register[j]
            eng.flush()
        eng.flush()

    def apply_stabilizer_circuit(self, eng, register, stabilizer):
        r"""
        Apply the stabilizer circuit to a ProjectQ Engine.

        Example: Applying "XYI" does X to first qubit register[0], and Y to second qubit
        register[1].

        Parameters
        ----------
        eng : BasicEngine
            The ProjectQ engine.
        register : list
            Holds the qubits.
        stabilizer : str or np.ndarray
            Either a pauli string representing one stabilier element or the binary representation
            of the one stabilizer element.

        Notes
        -----
        - Engine is flushed after.

        Examples
        --------
        >> eng = Project Q engine
        >> register = Qubits of Register
        >> apply_stabilizer_circuit(eng, register, "XXY")

        """
        if isinstance(stabilizer, (list, np.ndarray)):
            # Convert to Pauli String.
            pauli_str = StabilizerCode.binary_rep_to_pauli_str(stabilizer)

        for i, pauli_op in enumerate(pauli_str):
            print(pauli_op)
            if pauli_op == "X":
                XGate() | register[i]
            elif pauli_op == "Z":
                ZGate() | register[i]
            elif pauli_op == "Y":
                # ZGate() | register[i]
                # XGate() | register[i]
                # YGate() | register[i]
                QubitOperator('Y' + str(i), 1.) | register
        eng.flush()

    def decoding_circuit(self, eng, register, add_ancilla_bits=False, deallocate_nqubits=False):
        r"""
        Construct the decoding circuit to map the n-qubit to its k-qubit state.

        Specifically, suppose |D>_k is the unencoded k-qubit state and |D>_n is the encoded n-qubit
        state. The decoding circuit turns the n-qubit state |D>_n \otimes |0,...,0> tensored with
        k, ancilla qubits to the state |0,..,0> \otimes |D>_k.

        Parameters
        ----------
        eng : BasicEngine
            ProjectQ engine.
        register : list
            List containing the qubits/register for the ProjectQ engine "eng".
        add_ancilla_bits : bool
            If True, it will add extra, ancilla k-qubit. If it is false, it is assumed that it was
            already added and included in 'eng' and 'register'.
        deallocate_nqubits : bool
            If True, at the end of decoding it will discard and delete the 'register' and will only
            have the k, ancilla qubits.

        Returns
        -------
        list :
            If deallocate_nqubits is false, it returns the original 'register' plus the ancilla
            register appended towards the end.
            If deallocate_nqubits is True.
                If add_ancilla_bits is True, it returns the ancilla register that was created.
                If add_ancilla_bits is False, it assumes the register has the ancilla bits and
                returns that.

        Notes
        -----
        - To construct the most optimal decoding circuit. The standard form for the stabilizer
                code needs to be constructed alongside the logical X operators.
        - The engine is flushed at the end.
        - Let r be the rank of the standard form for the stabilizer code. This decoding scheme is
            more efficient when 2k(r + 1) < nr then just reversing the encoding circuit.

        References
        ----------
        - See Gaitan book "Quantum Error-Correction and Fault Tolerant Quantum Computing."

        """
        if add_ancilla_bits:
            register_ancilla = eng.allocate_qureg(self.k)
        else:
            register_ancilla = register[self.n:]
        logical_x = self.logical_x
        logical_z = self.logical_z

        # Turn |D>|0,...,0> to |D>|D>.
        for i_ancilla in range(0, self.k):
            logical_z_ith = logical_z[i_ancilla]
            for j_qubit, binary_val in enumerate(logical_z_ith[self.n:]):
                if binary_val == 1:
                    CNOT | (register[j_qubit], register_ancilla[i_ancilla])

        # Turn |D>|D> to |0,...,0>|D>
        for i_ancilla in range(0, self.k):
            logical_x_ith = logical_x[i_ancilla]

            with Control(eng, register_ancilla[i_ancilla]):
                pauli_str = StabilizerCode.binary_rep_to_pauli_str(logical_x_ith)
                for j_qubit, pauli_op in enumerate(pauli_str):
                    if pauli_op == "X":
                        XGate() | register[j_qubit]
                    elif pauli_op == "Y":
                        # YGate() | register[j_qubit]
                        ZGate() | register[j_qubit]
                        XGate() | register[j_qubit]
                        # QubitOperator('Y' + str(j_qubit), -1.j) | register
                    elif pauli_op == "Z":
                        ZGate() | register[j_qubit]

        eng.flush()
        if deallocate_nqubits:
            if add_ancilla_bits:
                All(Measure) | register
                del register
            else:
                All(Measure) | register[:self.n]
                register_ancilla = register[self.n:]
                del register[:self.n]
            return register_ancilla
        return register + register_ancilla

    def logical_circuit(self, eng, register, pauli_op):
        r"""
        Apply the circuit for the logical operator.

        Parameters
        ----------
        eng : BasicEngine

        register : list

        pauli_op : str
            String of the format "Xi", "Yi", "Zi", where i is a integer from 0 to k-1. E.g.
            'X0' will apply the 0th logical X-operator. There are a total of k logical X operators
            and k logical Z operators.

        Notes
        -----
        - The engine will be flushed at the end.

        Examples
        --------
        >> eng = ProjectQ engine...
        >> register = List of qubits in the engine.
        Apply the logical X operator on the first qubit.
        >> code.logical_circuit(eng, register, "X0")
        Apply the logical Z operator on the second qubit.
        >> code.logical_circuit(eng, register, "Z1"0

        """
        assert len(pauli_op) == 2, "'pauli_op' should be length two."
        assert pauli_op[0] in ['X', 'Y', 'Z'], "First character of 'pauli_op' should be X, Y or Z."
        assert 0 <= int(pauli_op[1]) < self.k, "Second character should be integer from 0 to k-1."

        if pauli_op[0] == 'X':
            logical_x_ith = self.logical_x[int(pauli_op[1])]
            pauli_str = StabilizerCode.binary_rep_to_pauli_str(logical_x_ith)
        elif pauli_op[0] == "Z":
            logical_z_ith = self.logical_z[int(pauli_op[1])]
            pauli_str = StabilizerCode.binary_rep_to_pauli_str(logical_z_ith)
        elif pauli_op[0] == 'Y':
            # The pauli Y is just ZX, which in binary representation is below.
            logical_y_ith = (self.logical_z[int(pauli_op[1])] +
                             self.logical_x[int(pauli_op[1])]) % 2
            pauli_str = StabilizerCode.binary_rep_to_pauli_str(logical_y_ith)

        for i, pauli_op in enumerate(pauli_str):
            if pauli_op == "X":
                XGate() | register[i]
            elif pauli_op == "Y":
                QubitOperator('Y' + str(i), 1.j) | register
            elif pauli_op == "Z":
                ZGate() | register[i]

    def undetectable_errors(self):
        r"""
        Undetectable errors N(S) - S of the stabilizer group S in pauli string.

        Returns
        -------
        list :
            Returns a list of strings of each pauli element in N(S) - S.

        """
        normalizer = self.normalizer()
        return [self._convert_binary_representation_to_pauli(x) for x in normalizer]

    def _gaussian_elimination_first_block(self):
        r"""
        Perform Gaussian Elimination On the first block G1 from binary representation [G1 | G2].

        Returns
        -------
        array :
            Returns the result of gaussian elimination performed on the binary representation of
            the first block G1.

        References
        ----------
        -- See Chapter Four in Frank Gaitan, "Quantum Error Correction and Fault Tolerent Quantum
            Computing".
        -- See Chapter 10 in Neilsen and CHaung.

        Notes
        -----
        -- Should produce a [I A | B C]
                            [0 0 | D E] matrix, where I is identity with rank of G1.

        """
        output = self.stab_bin_rep.copy()

        rank = np.linalg.matrix_rank(self.stab_bin_rep[:, :self.n])
        numb_stabs = self.numb_stab
        assert self.stab_bin_rep.shape[0] == numb_stabs

        # Perform Gaussian elimination by going through each column up to rank.
        for j in range(0, rank):
            # If it is zero, Need to swap diagonal row or column.
            if output[j, j] == 0:
                # Swap with a diagonal column!
                did_not_swap = True
                for i_col in range(j, self.n):
                    for j_row in range(j, numb_stabs):
                        if output[j_row, i_col] == 1 and did_not_swap:
                            # Swap with respect to X component of G1 binary representation [G1 | G2]
                            copy = output[:, i_col].copy()
                            output[:, i_col] = output[:, j]
                            output[:, j] = copy

                            # Swap same part for G2
                            copy = output[:, i_col + self.n].copy()
                            output[:, i_col + self.n] = output[:, j + self.n]
                            output[:, j + self.n] = copy

                            # Swap row too.
                            copy = output[j, :].copy()
                            output[j, :] = output[j_row, :]
                            output[j_row, :] = copy
                            did_not_swap = False

            # If it is still zero. Return a warning.
            if output[j, j] == 0:
                warnings.warn("Gaussian Elimination on Binary Representation Is Not Going to "
                              "Work.  It could not find a row or column replacement to make sure "
                              "diagonal element is one.")

            # Turn everything above and below the diagonal element to become zero.
            for i in range(0, numb_stabs):
                if i != j and output[i, j] != 0:
                    output[i, :] = (output[j, :] + output[i, :]) % 2

        return output, rank

    def _gaussian_elimination_second_block(self, binary_rep, rank):
        r"""
        Perform Gaussian Elimination of the Z component of binary representation.

        The binary_rep provided must have already performed _gaussian_elimination_first_block.

        Parameters
        ----------
        binary_rep : np.ndarray
            Binary representation where gaussian elimination was performed on G1 in [G1 | G2].
            G1 was already have gaussian elimination been attempted on it from the function,
            "_gaussian_elimination_first_block".

        Returns
        -------
        tuple :
            Tuple (A2, A2, B, C1, C2, D, E) of two-dimensional arrays from the standard normal
            form of,
                [[I A1 A2 | B C1 C2]
                [0 0 0   | D I E]]

        Notes
        -----
        - Unlike "_gaussian_elimination_first_block", this does not have sophicasted swapping
        (ie being able to swap diagonally). It only attempts to swap the based on row then
        based on columns.

        """
        if rank == binary_rep.shape[0]:
            return binary_rep

        output = binary_rep.copy()
        rank_E = np.linalg.matrix_rank(output[rank:, self.n + rank:])

        # Swap rows to ensure the diagonal elements are all ones.
        # TODO: Add More "Powerful" Swaps like in _gaussian_elimination_first_block
        for j in range(self.n + rank, self.n + rank + rank_E):
            if output[j - self.n, j] == 0:

                for i in range(j - self.n + 1, self.numb_stab):
                    # Find a row below it to swap below it!
                    if output[i, j] == 1:
                        copy = output[i, :].copy()
                        output[i, :] = output[j - self.n, :]
                        output[j - self.n, :] = copy
                        break

        # Perform Gaussian elimination by going through each column up to rank.
        for j in range(self.n + rank, self.n + rank + rank_E):
            diag_elem_E = output[j - self.n, j]  # Diagonal element of E.

            # If diagonal element is zero.
            if diag_elem_E == 0:
                # Find column with a one in diagonal and swap with it.
                for i_col in range(j + 1, 2 * self.n):
                    if output[j - self.n, i_col] == 1:
                        # Swap with respect to X component of G1 binary representation [G1 | G2]
                        copy = output[:, i_col].copy()
                        output[:, i_col] = output[:, j]
                        output[:, j] = copy

                        # Swap same part for G2
                        copy = output[:, i_col - self.n].copy()
                        output[:, i_col - self.n] = output[:, j - self.n]
                        output[:, j - self.n] = copy
                        break

            # If it is still zero. Return a warning.
            if diag_elem_E == 0:
                warnings.warn("Gaussian Elimination on Binary Representation Is Not Going to "
                              "Work.  It could not find a row or column replacement to make sure "
                              "diagonal element is one.")

            # Turn everything above and below the diagonal element to become zero.
            for i in range(rank, self.n - self.k):  # Go through rows.
                if i != j - self.n and output[i, j] != 0:
                    output[i, :] = (output[j - self.n, :] + output[i, :]) % 2
        return output

    def _standard_normal_form(self):
        gaus_elim, rank = self._gaussian_elimination_first_block()
        return self._gaussian_elimination_second_block(gaus_elim, rank), rank

    def _matrix_blocks_standard_form(self, standard_form, rank):
        r"""
        Given a standard form, obtain the block matrices A_2, E, C_1, C_2.

        Standard form is  [[I, A_1, A_2, B, C_1, C_2],
                            [0, 0, 0,    D, I, E]]

        Returns
        -------
        (array, array, array, array) :
            Returns the matrices respectively A_2, E, C_1, C_2

        """
        # Do assertions that the first block is the identity.
        assert np.all(np.abs(np.eye(rank, dtype=np.int) - standard_form[:rank, :rank]) < 1e-5), \
            "The standard form should have identity matrix in the top-left."

        a2 = standard_form[:rank, self.n - self.k: self.n]
        c1 = standard_form[:rank, self.n + rank: 2 * self.n - self.k]
        c2 = standard_form[:rank, 2 * self.n - self.k:]
        e = standard_form[rank:, 2 * self.n - self.k:]
        return a2, e, c1, c2

    def logical_operators(self):
        r"""
        Obtain the binary representation of the logical operators retrieved from the standard form.

        The logical operators are in block-form based on the block-matrices A_2, E, C_1,
        C_2 from the standard normal form.
            X = [0, E^T, I | E^TC_2^T + C_2^T, 0, 0]
            Z = [0 | A_2^T, 0, I]

        Returns
        -------
        list of numpy arrays :
            returns a tuple (X, Z), where X is the binary representation of logical operator X and
            Z is the binary representation of logical operator Z.

        References
        ----------
        See Frank Gaitan's book, chapter 4.

        """
        # Turn binary representation of stabilizer code into standard normal form.
        block, rank = self.normal_form, self.rank

        # obtain the block matrices from standard normal form.
        a2, e, c1, c2 = self._matrix_blocks_standard_form(block, rank)

        # Construct the logical X binary representation
        pad_zeros = np.zeros((self.k, rank), dtype=np.int)
        identity = np.eye(self.k, self.k, dtype=np.int)
        pad_zeros2 = np.zeros((self.k, self.n - rank), dtype=np.int)
        logical_X = np.hstack((pad_zeros, e.T, identity, (e.T.dot(c1.T) + c2.T) % 2, pad_zeros2))

        # Construct the logical Z binary representation
        pad_zeros = np.zeros((self.k, self.n), dtype=np.int)
        pad_zeros2 = np.zeros((self.k, self.n - self.k - rank), dtype=np.int)
        identity = np.eye(self.k, self.k, dtype=np.int)
        logical_Z = np.hstack((pad_zeros, a2.T, pad_zeros2, identity))

        return logical_X, logical_Z

    @staticmethod
    def inner_prod(pauli1, pauli2):
        r"""
        Return the sympletic inner-product on binary representation of two pauli elements modulo 2.

        Parameters
        ----------
        pauli1: np.ndarray
            Binary Representation of pauli element.
        pauli2: np.ndarray
            Binary Representation of pauli element.

        Returns
        -------
        int :
            Zero indicating commuting, One indiciating anti-commuting.

        """
        n = len(pauli1) // 2
        return (pauli1[:n].dot(pauli2[n:]) + pauli1[n:].dot(pauli2[:n])) % 2

    def generator_set_pauli_elements(self):
        r"""Return the set of all generators of pauli elements, without the identity element."""
        # TODO: Apprently the generators is just the identity matrix.
        # All combinations in the x-components of the binary representation.
        x_comps = np.zeros((self.n, self.n), dtype=np.int)
        zeros = x_comps.copy()
        # For each generator update the ith element to one.
        for i in range(self.n):
            x_comps[i][i] += 1
        # Symmetry in X and Z components.
        return np.vstack((np.hstack((x_comps, zeros)), np.hstack((zeros, x_comps))))

    def normalizer(self):
        r"""
        Get Normalizer N(S) - S of the stabilizer group S minus the stabilizer group.

        Note that this does not always return the generators and is a very slow algorithm.

        Returns
        -------
        array :
            Binary Representation of each N(S)-S element.

        """
        normalizer = []
        # Worse-case algorithm. Go through every permutation
        all_pauli_reps = itertools.product([0, 1], repeat=2 * self.n)
        # all_pauli_reps = self.generator_set_pauli_elements()
        for pauli in all_pauli_reps:
            pauli = list(pauli)
            does_commute = True

            # TODO: Vectorize this.
            for stab in self.stab_bin_rep:
                if not self._commute(stab, pauli):
                    does_commute = False
                    break
            # If it commutes and satisfies couple of generator conditions.
            if does_commute:
                normalizer.append(pauli)
        return np.array(normalizer)

    @staticmethod
    def concatenate_codes(stab1, stab2):
        r"""
        Concatenate two stabilizer codes S1 and S2 to create a code "S2 composed S1", with S1 first.

        Code concatenation only works when l divides n.

        Parameters
        ----------
        stab1: StabilizerCode
            The first/initial [n, k] stabilizer code.

        stab2: StabilizerCode
            The second/final [m, l] stabilizer code.

        encoder : bool
            True, then return the kraus operators for the encoder of the concantenated code.

        Returns
        -------
        StabilizerCode
            Returns stabilizer code for the new concatenated code.

        Raises
        ------
        ValueError :
            Code concatenation only works when l divides n.

        """
        assert isinstance(stab1, StabilizerCode)
        assert isinstance(stab2, StabilizerCode)

        n, k = stab1.n, stab1.k
        m, l = stab2.n, stab2.k

        if n % l != 0:
            assert ValueError("Code concatenation only works when l divides n, ie n | l.")

        generator = []
        # Suppose l = 1
        if l == 1:
            # Needs nm - l number of stabilizers to create a [nm, k] code.
            numb_blocks = n
            # Go through each block and copy the generators of stab2 to each block.
            for i in range(0, numb_blocks):
                for j in range(0, stab2.numb_stab):
                    new_gen_x = [0] * (i * m)
                    new_gen_z = [0] * (i * m)

                    new_gen_x.extend(stab2.stab_bin_rep[j, :m])
                    new_gen_z.extend(stab2.stab_bin_rep[j, m:])

                    new_gen_x += [0] * ((numb_blocks - i - 1) * m)
                    new_gen_z += [0] * ((numb_blocks - i - 1) * m)

                    generator.append(new_gen_x + new_gen_z)

            # Go through each generator from stab1.
            logical_x = stab2.logical_x[0]
            logical_z = stab2.logical_z[0]
            for i in range(0, stab1.numb_stab):
                stabilizer = stab1.stab_bin_rep[i]
                new_gen_x = []
                new_gen_z = []
                for j, pauli in enumerate(stabilizer[:n]):  # Go through each stabilizer.
                    if pauli == 1 and stabilizer[n + j] == 0:  # X Operator.
                        new_gen_x.extend(logical_x[:m])
                        new_gen_z += [0] * m
                    elif pauli == 0 and stabilizer[n + j] == 1:  # Z Operator.
                        new_gen_x += [0] * m
                        new_gen_z.extend(logical_z[m:])
                    elif pauli == 1 and stabilizer[n + j] == 1:  # Y Operator.
                        new_gen_x.extend(logical_x[:m])
                        new_gen_z.extend(logical_z[m:])
                    else:
                        new_gen_x += [0] * m
                        new_gen_z += [0] * m
                generator.append(new_gen_x + new_gen_z)
            return StabilizerCode(generator, n * m, k)

        if (n % l) == 0:
            # Divide
            numb_blocks = n // l
            total_qubits = numb_blocks * m

            for i in range(0, numb_blocks):  # Go Through Each Block
                for j in range(0, stab2.numb_stab):  # Copy the Stabilizers of Code 2 to each Block.
                    new_gen_x = [0] * (i * m)
                    new_gen_z = [0] * (i * m)

                    new_gen_x.extend(stab2.stab_bin_rep[j, :m])
                    new_gen_z.extend(stab2.stab_bin_rep[j, m:])

                    new_gen_x += [0] * ((numb_blocks - i - 1) * m)
                    new_gen_z += [0] * ((numb_blocks - i - 1) * m)

                    generator.append(new_gen_x + new_gen_z)
            # Go through each generator from stab1.
            logical_x = stab2.logical_x
            logical_z = stab2.logical_z
            for i in range(0, stab1.numb_stab):
                stabilizer = stab1.stab_bin_rep[i]
                new_gen_x = np.zeros((total_qubits), dtype=np.int)
                new_gen_z = np.zeros((total_qubits), dtype=np.int)

                for j, pauli in enumerate(stabilizer[:n]):  # Go through each stabilizer element.
                    which_logical_op = j % numb_blocks  # Modulo the number of blocks.

                    which_block = int(np.floor(j / numb_blocks))

                    if pauli == 1 and stabilizer[n + j] == 0:  # X Block Operator.
                        current_logical = logical_x[which_logical_op]
                    elif pauli == 0 and stabilizer[n + j] == 1:  # Z Block Operator
                        current_logical = logical_z[which_logical_op]
                    elif pauli == 1 and stabilizer[n + j] == 1:  # Y Block Operator
                        current_logical = (logical_z[which_logical_op] +
                                           logical_x[which_logical_op]) % 2

                    # Get the range of qubits that it acts on.
                    # TODO: should this be n or m.
                    range_qubits = (n * which_block, n * (which_block + 1))

                    new_gen_x[range_qubits[0]: range_qubits[1]] = \
                        (new_gen_x[range_qubits[0]: range_qubits[1]] + current_logical[:m]) % 2
                    new_gen_z[range_qubits[0]: range_qubits[1]] = \
                        (new_gen_z[range_qubits[0]: range_qubits[1]] + current_logical[m:]) % 2

                generator.append(np.hstack((new_gen_x, new_gen_z)))

            return StabilizerCode(np.array(generator), total_qubits, k)

    @staticmethod
    def binary_rep_to_pauli_str(binary_rep):
        r"""
        Convert a binary representation of single stabilizer to a pauli string.

        Parameters
        ----------
        binary_rep : (list(2n) or np.ndarray(2n,))
            List or one-dimensional numpy array containing 2*n binary elements, representing the
            binary representation of a single stabilizer element.

        Returns
        -------
        str :
            Returns the string of pauli matrices representing the binary representation.

        Examples
        --------
        binary_rep = np.array([0, 0, 1, 1, 0, 0])
        pauli_str = StabilizerCode.binary_rep_to_pauli_str(binary_rep)
        #  Should be "ZIX".

        """
        assert isinstance(binary_rep, (list, np.ndarray))
        pauli = ""
        n = len(binary_rep) // 2
        for i, x in enumerate(binary_rep[:n]):
            z = binary_rep[n + i]
            if x == 1 and z == 1:
                pauli += "Y"
            elif x == 0 and z == 0:
                pauli += "I"
            elif x == 1:
                pauli += "X"
            elif z == 1:
                pauli += "Z"
            else:
                raise RuntimeError("Binary Representation is not correct.")
        return pauli

    @staticmethod
    def binary_rep_to_pauli_mat(binary_rep, sparse=False):
        r"""
        Convert binary representations of stabilizer code to set of pauli matrices.

        Parameters
        ----------
        binary_rep : array
            Two-dimensional array where rows correspond to binary representation of stabilizer
            element.

        sparse : bool
            True, if pauli-matrices are returned sparsed.

        Returns
        -------
        list :
            Returns a set of pauli-matrices. They are sparse if sparse is true.

        """
        if binary_rep.ndim != 2:
            raise TypeError("Binary Representation needs to be two-dimensional.")

        operators = []
        n = binary_rep.shape[1] // 2  # Number of qubits.
        # Go through each stabilizer element.
        for i, stab in enumerate(binary_rep):
            # Convert the first column to pauli-matrix.
            pauli = StabilizerCode.index_to_qubit_pauli(stab[0], stab[n], sparse)

            # Go through each individual pauli-qubit.
            for j in range(1, n):
                x_j, z_j = stab[j], stab[j + n]
                pauli_mat = StabilizerCode.index_to_qubit_pauli(x_j, z_j, sparse)
                if sparse:
                    pauli = kron(pauli, pauli_mat, format="csr")
                else:
                    pauli = np.kron(pauli, pauli_mat)
            operators.append(pauli)
        return operators

    @staticmethod
    def pauli_str_to_binary_representation(paulis):
        r"""
        Binary representation of the pauli-operators.

        Parameters
        ----------
        paulis : list
            List of strings of pauli elements.

        Returns
        -------
        np.ndarray :
            Returns a two-dimensional array corresponding to the binary representation of the
            pauli operators.

        """
        n = len(paulis[0])
        binary_rep = np.zeros((len(paulis), 2 * n))
        for i, p in enumerate(paulis):
            for j in range(0, n):
                if p[j] == "X":
                    binary_rep[i][j] = 1
                elif p[j] == "Z":
                    binary_rep[i][n + j] = 1
                elif p[j] == "Y":
                    binary_rep[i][j] = 1
                    binary_rep[i][j + n] = 1
                elif p[j] == "I":
                    pass  # Do Nothing
                else:
                    raise ValueError("Pauli string symbol not recognized: " + str(p[j]))
        return binary_rep

    @staticmethod
    def index_to_qubit_pauli(x1, x2, sparse=False):
        r"""
        Given two indices returns the pauli matrix as a numpy array.

        Parameters
        ----------
        x1 : int
            The index that tells whether it is a X matrix or not.
        x2 : int
            The index that tells whether it is a Z matrix or not.
        sparse : bool
            If true, returns the pauli matrix {X, Y, Z or I} in sparse matrix format.

        Returns
        -------
        np.ndarray or CSR_Matrix
            Either returns the single qubit pauli matrix as a numpy array or a sparse csr matrix.

        """
        x, y, z, i = X, Y, Z, I
        if sparse:
            x, y, z, i = XS, YS, ZS, IS
        if x1 == 1 and x2 == 0:
            return x
        elif x1 == 1 and x2 == 1:
            return y
        elif x1 == 0 and x2 == 1:
            return z
        elif x1 == 0 and x2 == 0:
            return i
        else:
            raise ValueError("Binary Representation is incorrect.")
