from abc import ABC, abstractmethod, abstractstaticmethod
from projectq.ops import CNOT, H, S, Measure
import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh
import itertools


__all__ = ["QCode", "StabilizerCode"]


# Pauli Operators Plus Sparse Pauli Operators
X, Y, Z, I = [np.array([[0., 1.], [1., 0.]]),
              np.array([[0., complex(0, -1.)], [complex(0., 1.), 0.]]),
              np.array([[1., 0.], [0, -1.]]), np.eye(2)]

XS, YS, ZS, IS = [csr_matrix(X, dtype=np.complex128),
                  csr_matrix(Y, dtype=np.complex128),
                  csr_matrix(Z, dtype=np.complex128),
                  identity(2, dtype=np.complex128, format="csr")]


class QCode(ABC):
    r"""
    Abstract base class for quantum codes.

    """
    def __init__(self, encoder, decoder, correction, n, k):
        pass

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass

    @abstractmethod
    def correct(self):
        pass

    @abstractmethod
    def optimize_encoder(self):
        pass

    @abstractmethod
    def optimize_decoder(self):
        pass

    @abstractstaticmethod
    def concatenate_codes(self):
        pass


class StabilizerCode():
    r"""
    Class for analyzing stabilizer codes.

    """
    def __init__(self, stabilizer_group, n, k, logical_x, logical_y):
        r"""

        stabilizer_group : array
            Binary Representation of stabilizer groups.

        logical_x : array, optional


        logical_y : array, optional


        n : int
            Number of total qubits.

        k : int
            Number of encoded qubits.
        """
        # Turn List into Numpy array
        if isinstance(stabilizer_group, list):
            stabilizer_group = np.array(stabilizer_group)
        self.stab_bin_rep = stabilizer_group
        assert self.stab_bin_rep.shape[0] <= n # Double check this...
        assert self.stab_bin_rep.shape[0] <= 4**n
        self.numb_stab = self.stab_bin_rep.shape[0]
        self.dim_code_space = n - self.numb_stab
        self.n = n
        self.k = k
        assert self._is_stabilizer_code()

    def __str__(self):
        # Convert binary representation to pauli elements.
        string = []
        for bin_stab in self.stab_bin_rep:
            string.append(StabilizerCode.binary_rep_to_pauli_str(bin_stab))
        return str(string)

    def _inner_prod(self, pauli1, pauli2):
        r"""
        Return the sympletic inner-product on binary representation of two pauli elements.

        Parameters
        ----------
        pauli1: array
            Binary Representation of pauli element.

        pauli2: array
            Binary Representation of pauli element.

        Returns
        -------
        int :
            Zero indicating commuting, One indiciating anti-commuting.
        """
        return pauli1[:self.n].dot(pauli2[self.n:]) + pauli1[self.n:].dot(pauli2[:self.n])

    def _is_stabilizer_code(self):
        r"""
        Check if all pauli elements in the group is commutative.

        Return
        ------
        bool
            True if pauli elements are commutative.
        """
        is_stabilizer_code = True
        for paul1, paul2 in zip(self.stab_bin_rep, self.stab_bin_rep):
            if not self._commute(paul1, paul2):
                is_stabilizer_code = False
                break
        return is_stabilizer_code

    def _eigenspace_one(self, pauli_sp, multi):
        # Get the eigenspace of eigenvalue one of Sparse Pauli operator.
        # Each pauli operator has eigenvalue one with multiplcity 2^(n-1)
        k = 2**self.n // 2
        eigs, evecs = eigsh(pauli_sp, multi, which="LM", sigma=1.0001, maxiter=100000)
        return eigs, evecs

    def encode_krauss_operators(self, sparse=False):
        r"""
        Isometry operator for the encoder.

        Parameters
        ----------
        sparse : bool
            True, return the encoder map as a sparse matrix.

        Returns
        -------
        np.array :
            Returns a numpy array representing the encoder map from k-qubits to n-qubits,
            given a (n, k)-parameter code.
        """
        def _obtain_eigenspace_one(eigen, vecs):
            return vecs[:, np.abs(eigen - 1) < 1e-5]

        # Diagonalize first stabilizer element and get only plus one eigenspace.
        stabilizers = StabilizerCode.binary_rep_to_pauli_mat(self.stab_bin_rep)
        stab1 = stabilizers[0]
        eigs, evecs = np.linalg.eigh(stab1)
        evecs = _obtain_eigenspace_one(eigs, evecs)

        # Continue Simutaneously Diagonalizing
        for i in range(1, self.numb_stab):
            conju = evecs.conjugate().T.dot(stabilizers[i].dot(evecs))
            e1, ev1 = np.linalg.eigh(conju)
            ev1 = _obtain_eigenspace_one(e1, ev1)
            evecs = evecs.dot(ev1)

        if sparse:
            return csr_matrix(evecs, shape=evecs.shape, dtype=np.complex128)
        return evecs

    def decode_krauss_operators(self, pauli_errors):
        r"""
        Kraus operators for the decoder.

        Parameters
        ----------
        pauli_errors : list of str
            List of strings representing pauli errors.

        Returns
        -------
        list :
            List of strings representing stabilizer element anti-commuting with the pauli
            errors.
        """
        # Find a list of stabilizer that anti-commutes with pauli_errors
        stabilizer = []

        binary_rep = StabilizerCode.pauli_str_to_binary_representation(pauli_errors)
        for error in binary_rep:
            for i, stab in enumerate(self.stab_bin_rep):
                if not self._commute(error, stab):
                    stabilizer.append(StabilizerCode.binary_rep_to_pauli_str(stab))
                    break
                if i == self.stab_bin_rep.shape[0] - 1:
                    raise ValueError("A error did not anti-commute with stabilizers.")
        return np.array(stabilizer)

    def syndrome_measurement(self, eng, register, stabilizer):
        r"""
        Get the syndrome measurement of stabilizer element.

        Parameters
        ----------
        eng : ProjectQ Engine

        register : ProjectQ Qureg or list of Qubit
            Either a quantum register or a list of qubits.

        stabilizer : (list, np.ndarray(2n,))
            Binary representation of the stabilizer element.

        Returns
        -------
        int :
            The measurement corresponding to the stabilizer element.
        """
        numb_qubits = len(eng.active_qubits)

        # The additional qubit is for measurement purposes.
        if numb_qubits != self.n:
            raise TypeError("Number of qubits allocated should match the number of encoded qubits n "
                            "from (n,k) code.")

        # Allocate a new qubit for measurement, if it doesn't have it already.
        if numb_qubits == self.n:
            measure_qubit = eng.allocate_qubit()

        # Convert stabilier element to pauli matrix and construct the circuit and measure it.
        pauli_mat = self.binary_rep_to_pauli_str(np.array(stabilizer))
        for i, pauli_element in enumerate(pauli_mat):
            if pauli_element == 'X':
                H | register[i]
                CNOT | (register[i], measure_qubit)
                H | register[i]
            elif pauli_element == 'Y':
                S | register[i]
                H | register[i]
                S | register[i]
            elif pauli_element == 'Z':
                CNOT | (register[i], measure_qubit)
            elif pauli_element == "I":
                pass
            else:
                raise RuntimeError("Pauli strings contained an unknown character.")

        Measure | measure_qubit
        # eng.flush()
        result = int(measure_qubit)
        del measure_qubit
        return result

    def encoding_circuit(self):
        pass

    def decoding_circuit(self):
        pass

    def _commute(self, binary_rep1, binary_rep2):
        inner_prod = self._inner_prod(binary_rep1, binary_rep2) % 2
        # If one then they anti-commute
        if inner_prod:
            return False
        return True

    def _generator_conditions(self, pauli):
        r"""
        Necessary conditions to satisfy for a pauli element to be a generator.
        """
        # Get X and Z Components of Pauli element.
        x_comps = pauli[:self.n]
        z_comps = pauli[self.n:]

        # Check if any of them contain a pauli Y.
        contains_y = np.any(np.abs(x_comps - z_comps) < 1e-5)
        # Check if all Y
        all_Y = np.all(np.abs(pauli - 1) < 1e-5)
        # Condition that they are not all X or all Y or all Z
        conditions = (not contains_y) and (not all_Y)
        return conditions

    def generator_set_pauli_elements(self):
        r"""
        Return the set of all generators of pauli elements, without the identity element.
        """
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
            if does_commute:# and self._generator_conditions(np.array(pauli)):
                normalizer.append(pauli)
        return np.array(normalizer)

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

    def correction(self):
        pass

    def _gaussian_elimination_first_block(self):
        r"""
        Perform Gaussian Elimination On the first block G1 from binary representation [G1 | G2]

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

        # Swap rows to ensure the diagonal elements are all ones.
        for j in range(0, rank):
            if output[j, j] == 0:
                for i in range(j + 1, numb_stabs):
                    # Find a row below it to swap below it!
                    if output[i, j] == 1:
                        copy = output[i, :].copy()
                        output[i, :] = output[j, :]
                        output[j, :] = copy
                        break

        # Perform Gaussian elimination by going through each column up to rank.
        for j in range(0, rank):

            # If diagonal element is zero.
            if output[j, j] == 0:
                # Swap with a column!
                for i_col in range(j + 1, self.n):
                    if output[j, i_col] == 1:
                        # Swap with respect to X component of G1 binary representation [G1 | G2]
                        copy = output[:, i_col].copy()
                        output[:, i_col] = output[:, j]
                        output[:, j] = copy

                        # Swap same part for G2
                        copy = output[:, i_col + self.n].copy()
                        output[:, i_col + self.n] = output[:, j + self.n]
                        output[:, j + self.n] = copy
                        break

            # Turn everything above and below the diagonal element to become zero.
            for i in range(0, numb_stabs):
                if i != j and output[i, j] != 0:
                    output[i, :] = (output[j, :] + output[i, :]) % 2

        return output, rank

    def _gaussian_elimination_second_block(self, binary_rep, rank):
        r"""

        Parameters
        ----------
        binary_rep : np.ndarray
            Binary representation where gaussian elimination was performed on G1 in [G1 | G2].

        Returns
        -------
        tuple :
            Tuple (A2, A2, B, C1, C2, D, E) of two-dimensional arrays from the standard normal
            form of,
                [[I A1 A2 | B C1 C2]
                [0 0 0   | D I E]]

        """
        if rank == binary_rep.shape[0]:
            return binary_rep

        output = binary_rep.copy()
        rank_E = np.linalg.matrix_rank(output[rank:, self.n + rank : ])

        # Swap rows to ensure the diagonal elements are all ones.
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

        """
        # Do assertions that the first block is the identity.
        assert np.all(np.abs(np.eye(rank, dtype=np.int) - standard_form[:rank, :rank]) < 1e-5), \
            "The standard form should have identity matrix in the top-left."

        # a1 = standard_form[:rank, rank : self.n - self.k]
        a2 = standard_form[:rank, self.n - self.k : self.n]
        # b = standard_form[:rank, self.n : self.n + rank]
        c1 = standard_form[:rank, self.n + rank : 2 * self.n - self.k]
        c2 = standard_form[:rank, 2 * self.n - self.k :]
        e = standard_form[rank:, 2 * self.n - self.k :]
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
        block, rank = self._standard_normal_form()

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
    def index_to_qubit_pauli(x1, x2, sparse=False):
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

    @staticmethod
    def concatenate_codes(stab1, stab2):
        r"""
        Concatenate two stabilizer codes S1 and S2 to create a code S1 composed S2.

        Parameters
        ----------
        stab1:
            A [n, k] stabilizer code.

        stab2:
            A [m, l] stabilizer code.

        encoder : bool
            True, then return the kraus operators for the encoder of the concantenated code.

        Returns
        -------
        StabilizerCode
            Returns stabilizer code.
        """
        assert isinstance(stab1, StabilizerCode)
        assert isinstance(stab2, StabilizerCode)

        n, k = stab1.n, stab1.k
        m, l = stab2.n, stab2.k

        # Suppose k = 1
        if k == 1:
            # Need a nm - l number of stabilizers.
            pass


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
            if x == 1 and z == 1: pauli += "Y"
            elif x == 0 and z == 0: pauli += "I"
            elif x == 1: pauli += "X"
            elif z == 1: pauli += "Z"
            else: raise RuntimeError("Binary Representation is not correct.")
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
        n = binary_rep.shape[1] // 2 # Number of qubits.
        # Go through each stabilizer element.
        for i, stab in enumerate(binary_rep):
            # Convert each column to pauli-matrix.
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
        """
        n = len(paulis[0])
        binary_rep = np.zeros((len(paulis), 2 * n))
        for i, p in enumerate(paulis):
            for j in range(0, n):
                if p[j] == "X":
                    binary_rep[i][j] = 1
                elif p[j] == "Z":
                    binary_rep[i][n + j] = 1
                elif [j] == "Y":
                    binary_rep[i][j] = 1
                    binary_rep[i][j + n] = 1
                else:
                    raise ValueError("Pauli string symbol not recognized.")
        return binary_rep
