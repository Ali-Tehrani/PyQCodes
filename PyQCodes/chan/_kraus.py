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
import numpy as np
from scipy.sparse import kron
from sparse import COO, kron, tensordot, matmul, SparseArray
from abc import ABC, abstractmethod, abstractstaticmethod

r"""
Kraus operators for representing Quantum Channels.

Classes
-------
- Kraus :
    Abstract Base Class
- DenseKraus :
    Kraus operators as dense matrices.
- SparseKraus :
    Kraus operators as sparse matrices.
    
The Functionality (what it can do) of DenseKraus is really similar to SparseKraus.
"""


__all__ = ["SparseKraus", "DenseKraus"]


class Kraus(ABC):
    r"""Abstract Base Class for kraus Operators."""

    def __init__(self, kraus_ops, numb_qubits, dim_in, dim_out):
        r"""
        Construct of Abstract Base Class for Kraus Operators.

        Parameters
        ----------
        kraus_ops : list, array
            List or three-dimensional numpy array containing all the kraus operators representing a
            quantum channel.
        numb_qubits : list
            Number of qubits [M, N] of input and output of the quantum channel.
        dim_in : int
            Dimension of a single-particle/qubit hilbert space. For qubit, dimension is two by
            definition.
        dim_out : int
            Dimension of a single-particle/qubit hilbert space of the output. For qubit, dimension
            is two by definition.

        Raises
        ------
        TypeError
            Number of columns of each kraus operators should match dim_in**numb_qubits.
            Number of rows of each kraus operators should match dim_out**numb_qubits.

        """
        # Test the input conditions.
        if not isinstance(kraus_ops, (list, np.ndarray, SparseArray)):
            raise TypeError("Kraus operators should be a list or numpy array or sparse array.")
        # Assert that kraus operators be three-dimensional.
        if isinstance(kraus_ops, np.ndarray):
            assert kraus_ops.ndim == 3, "kraus operators should be three-dimensional numpy array."
        assert isinstance(numb_qubits, list), "Number of qubits should be an list."
        assert isinstance(numb_qubits[0], int), "Elements of numb_qubits should be Integer."
        assert isinstance(numb_qubits[1], int), "Elements of numb_qubits should be Integer."
        assert len(numb_qubits) == 2, "Length of numb_qubits should be two."
        assert isinstance(dim_in, int), "Dimension of Input Hilbert space should be integer."
        assert isinstance(dim_out, int), "Dimension of Output Hilbert space should be integer."
        # Test dimensions match each kraus operator shape.
        for k in kraus_ops:
            assert k.shape[1] == dim_in**numb_qubits[0], "Number of Column of kraus operators " \
                                                         "should match dims specified."
            assert k.shape[0] == dim_out**numb_qubits[1], "Number of Rows of kraus " \
                                                          "operators should match dims specified."

        self.numb_kraus = len(kraus_ops)
        # Note that numb_qubits reset to one when multiplying kraus operators A * B, see __mul__.
        self._numb_qubits = numb_qubits
        self.input_dim = dim_in**numb_qubits[0]
        self.output_dim = dim_out**numb_qubits[1]
        self._dim_in = dim_in
        self._dim_out = dim_out

        # Default n is always one.
        self.current_n = 1
        self._shape = (dim_in**numb_qubits[0], dim_out**numb_qubits[1])

        # Array for updating the kraus operators for the nth-channel use.
        self._nth_kraus_ops = None
        self._nth_kraus_ops_conj = None  # Only for Sparsekraus
        self._initialize_kraus_ops()

        # Array needed for computing entropy exchange.
        self._nth_kraus_exch = None
        self._kraus_operators_exchange(self.current_n)

    def __len__(self):
        # Number of Kraus Operators.
        return self.numb_kraus

    @property
    def kraus_ops(self):
        return self._kraus_ops

    @kraus_ops.setter
    def kraus_ops(self, kraus):
        self._kraus_ops = kraus

    @property
    def numb_qubits(self):
        return self._numb_qubits

    @property
    def dim_in(self):
        return self._dim_in

    @property
    def dim_out(self):
        return self._dim_out

    @property
    def nth_kraus_ops(self):
        return self._nth_kraus_ops

    @nth_kraus_ops.setter
    def nth_kraus_ops(self, kraus):
        self._nth_kraus_ops = kraus

    @property
    def nth_kraus_ops_conj(self):
        return self._nth_kraus_ops_conj

    @nth_kraus_ops_conj.setter
    def nth_kraus_ops_conj(self, kraus):
        self.nth_kraus_ops_conj = kraus

    @property
    def nth_kraus_exch(self):
        # Return kraus operators for computing entropy exchange.
        return self._nth_kraus_exch

    @abstractmethod
    def __add__(self, other):
        r"""
        Return Serial Concatenation of two kraus operators B and A, where A is applied first then B.

        Parameters
        ----------
        other : Kraus
            Kraus operator object of the channel being concatenated B \circ A.

        Returns
        -------
        Kraus :
            Kraus operator with first applying kraus operators A then applying kraus operators B.

        Examples
        --------
        # Serial Concatenate Identity Channel then Dephasing Channel
        numb_qubits, dim_in, dim_out = [1, 1], 2, 2
        identity = Kraus([np.array([[1., 0.], [0., 1.]]), numb_qubits, dim_in, dim_out)

        # Dephasing Channel
        k0 = np.sqrt(0.5) * np.array([[1., 0.], [0., 1.]])
        k1 = np.sqrt(0.5) * np.array([[1., 0.], [0., -1.]]
        kraus = [k0, k1]
        dephas = Kraus(kraus, numb_qubits, dim_in, dim_out)

        # New channel with Input -> Identity -> Dephasing -> Output
        new_kraus = dephas + identity
        # Should be equivalent to dephasing channel.

        """
        pass

    @abstractmethod
    def __mul__(self, other):
        r"""
        Parallel Concatenation (tensor of channels) of two kraus operators sets A and B.

        Parameters
        ----------
        other : Kraus
            Kraus operator object of the channel being concatenated A \otimes B.

        Returns
        -------
        Kraus :
            Kraus operator object representing A \otimes B with A is applied to first qubit and
            B is applied to second qubit.

        Notes
        -----
        The number of particles/qubits is reset to 1 to 1, for programming purposes.

        Examples
        --------
        # Parallel Concatenate Dephasing Channel with Identity
        nnumb_qubits, dim_in, dim_out = [1, 1], 2, 2
        identity = Kraus([np.array([[1., 0.], [0., 1.]]), numb_qubits, dim_in, dim_out)

        # Dephasing Channel
        k0 = np.sqrt(0.5) * np.array([[1., 0.], [0., 1.]])
        k1 = np.sqrt(0.5) * np.array([[1., 0.], [0., -1.]]
        kraus = [k0, k1]
        dephas = Kraus(kraus, numb_qubits, dim_in, dim_out)

        # New channel with Input -> Identity -> Dephasing -> Output
        new_kraus = dephas * identity
        # Should be equivalent to applying dephasing channel on first qubit and identity on
        # second qubit.

        """
        pass

    def update_kraus_operators(self, n):
        r"""
        Update the kraus operators to represent the channel tensored n-times.

        Parameters
        ----------
        n : int
            The channel A tensored by itself n times A^{\otimes n}.

        Examples
        --------
        # Dephasing Channel when n = 1, but want n = 2
        n, dim_in, dim_out = 1, 2, 2
        k0 = np.sqrt(0.5) * np.array([[1., 0.], [0., 1.]])
        k1 = np.sqrt(0.5) * np.array([[1., 0.], [0., -1.]]
        kraus = [k0, k1]
        kraus_obj = Kraus(kraus, n, dim_in, dim_out)
        kraus_obj.update_kraus_operators(n=2)
        # Kraus operators when n=2 should be
        # [k0 \otimes k0, k0 \otimes k1, k1 \otimes k0, k1 \otimes k2]

        # To revert back to n = 1
        kraus_obj.update_kraus_operators(n=1)
        # Kraus operators when n=1 should be [k0, k1]

        """
        if self.current_n < n:
            # Update Densekraus operator to correspond to nth channel
            for i in range(self.current_n, n):
                self.nth_kraus_ops = self._tensor_kraus_operators(self.kraus_ops,
                                                                  self.nth_kraus_ops)
        elif n < self.current_n:
            # Reduce Densekraus operator to correspond to nth channel.
            if n == 1:
                self.nth_kraus_ops = self.kraus_ops
            else:
                self.nth_kraus_ops = self._tensor_kraus_operators(self.kraus_ops, self.kraus_ops)
                for i in range(0, n - 2):
                    self.nth_kraus_ops = self._tensor_kraus_operators(self.kraus_ops,
                                                                       self.nth_kraus_ops)
        # Update current n.
        self.current_n = n
        # Update the matrix for computing the entropy-exchange matrix.
        self._kraus_operators_exchange(n)

    @abstractmethod
    def _tensor_kraus_operators(self, array1, array2):
        # Tensor a two sets of kraus operators together. Same as Parallel Concatenation of two
        # channels.
        pass

    @abstractmethod
    def _kraus_operators_exchange(self, n):
        # Update the matrix needed for entropy exchange matrix
        pass

    @abstractmethod
    def entropy_exchange(self, rho, n, adjoint=False):
        # Compute the entropy exchange matrix
        pass

    @abstractmethod
    def _initialize_kraus_ops(self):
        # Initialize kraus operators to help other methods for the class.
        pass

    @abstractstaticmethod
    def serial_concatenate(kraus1, kraus2):
        # Serial Concatenate Two Set of Kraus Operators Together.
        pass

    @abstractstaticmethod
    def parallel_concatenate(kraus1, kraus2):
        # Parallel Concatenate Two Set of Kraus Operators Together.
        pass


class SparseKraus(Kraus):
    r"""
    Kraus operators represented as Sparse matrices from scipy package.

    Attributes
    ----------
    kraus_ops : array
        Three-dimensional Coordinate format (COO) array holding the initial (n=1) kraus operators.
    current_n : int
        How many times the channel A is tensored A^{\otimes n}.
    nth_kraus_ops : array
        Three-dimensional COO array holding kraus operators representing the channel tensored
        "current_n" times.

    Methods
    -------
    is_trace_perserving :
        Return true if the kraus operators represent a trace-perserving map.
    update_kraus_operators :
        Update the kraus operators to correspond to the channel tensored n times.
    channel(rho) :
        Apply the channel to a hermitian matrix rho.
    entropy_exchange(rho) :
        Apply the complementary channel to a hermitian matrix rho.
    average_entanglement_fidelity :
        Calculate the average entanglement fidelity with respect to an ensemble of states.
    parallel_concatenate :
        Given two channels A and B, return kraus operators representing "A tensored B".
    serial_concatenate :
        Given two channels A and B, return kraus operators representing "A composed B".

    Examples
    ---------
    # Construct Bit-Flip Map and apply to a channel.
    X = np.array([[0., 1.], [1., 0.]])
    I = np.array([[1., 0.], [0., 1.]])
    prob_error = 0.25
    kraus_ops = [np.sqrt(1 - prob_error) * I, np.sqrt(prob_error) * X]
    numb_qubits, dim_in, dim_out = [1, 1], 2, 2
    kraus = SparseKraus(kraus_ops, numb_qubits, dim_in, dim_out)
    rho = (I + 0.2 * X) / 2.
    channel_rho = DenseKraus.channel(rho, n=1)

    # Dephrasure Channel (Orthogonal Kraus Example)
    numb_qubits, dim_in = [1, 1], 2
    dim_out = 3
    Z = np.array([[1., 0.], [0., -1.], [0., 0.]])  # Kraus operator for dephasing channel.
    e1 = np.array([[0., 0.], [0., 0.], [0., 1.]])  # Kraus operators for erasure channel.
    e2 = np.array([[0., 0.], [0., 0.], [1., 0.]])  # Kraus operators for erasure channel.
    p_dep = 0.25  # Probability dephasing error
    p_era = 0.5  # Probability of erasure error
    kraus_ops = [np.sqrt(1 - p_era) * np.sqrt(1 - p_dep) * I,
                 np.sqrt(1 - p_era) * np.sqrt(p_dep) * Z,
                 np.sqrt(p_era) * e1, np.sqrt(p_era) * e2]
    ortho_kraus = [2]  #  Index of Erasure kraus in "kraus_ops" are orthogonal to dephasing ops.
    kraus = SparseKraus(kraus_ops, numb_qubits, dim_in, dim_out, orthogonal_kraus)

    # Apply to dephrasure channel tensored two times on a diagonal two-qubit density matrix.
    rho = np.diag([0.25, 0.25, 0.25, 0.25])
    channel = kraus.channel(rho, n=2)

    """

    def __init__(self, kraus_ops, numb_qubits, dim_in=2, dim_out=2):
        r"""
        Construct for the kraus operators represented as sparse matrices.

        Parameters
        ----------
        kraus_ops : list, array, SparseArray
            List containing all the kraus operators representing a quantum channel.
        numb_qubits : list
            Number of particles/qubits [M, N] of the input and output of the quantum channel.
        dim_in : int
            Dimension of a single-particle/qubit hilbert space. For qubit, dimension is two by
            definition.
        dim_out : int
            Dimension of a single-particle/qubit hilbert space of the output. For qubit, dimension
            is two by definition.

        """
        # If kraus operators aren't in sparse format.
        if isinstance(kraus_ops, (list, np.ndarray)):
            self.kraus_ops = COO(np.array(kraus_ops, dtype=np.complex128),)
        else:
            self.kraus_ops = kraus_ops
        super(SparseKraus, self).__init__(kraus_ops, numb_qubits, dim_in, dim_out)

    def __add__(self, other):
        r"""Return Sparse Kraus from serial concatenation of two kraus operators."""
        assert isinstance(other, SparseKraus), "Kraus object should be of type 'SparseKraus'."
        if self.dim_in**self.numb_qubits[0] != other.dim_out**other.numb_qubits[1]:
            raise TypeError("Kraus operator dimension do not match.")
        numb_qubits = [self.numb_qubits[0], other.numb_qubits[1]]
        dim_in, dim_out = other.dim_in, self.dim_out
        return SparseKraus(SparseKraus.serial_concatenate(self.kraus_ops, other.kraus_ops),
                           numb_qubits, dim_in, dim_out)

    def __mul__(self, other):
        r"""Return Sparse Kraus from parallel concatenation of two kraus operators."""
        assert isinstance(other, SparseKraus), "Kraus object should be of type 'SparseKraus'."
        # Assert that dimensions of kraus operators match each other.
        # The number of qubits is reset to one.
        numb_qubits = [1, 1]
        dim_in, dim_out = other.dim_in * self.dim_in, other.dim_out * self.dim_out
        return SparseKraus(SparseKraus.parallel_concatenate(self.kraus_ops, other.kraus_ops),
                           numb_qubits, dim_in, dim_out)

    def _initialize_kraus_ops(self):
        r"""Initialize kraus operators for computation of channel and entropy exchange."""
        self._nth_kraus_ops = self.kraus_ops.copy(deep=True)
        self._nth_kraus_ops.enable_caching()
        self._nth_kraus_ops_conj = self.nth_kraus_ops.conj().transpose((0, 2, 1))
        self._nth_kraus_ops_conj.enable_caching()

    def is_trace_perserving(self):
        r"""Return true if kraus operators are trace-perserving."""
        length = self.kraus_ops[0].shape[1]
        trace_cond = np.zeros((length, length), dtype=np.complex128)
        for k in self.kraus_ops:
            trace_cond += k.conj().T.dot(k).todense()
        return np.all(np.abs(trace_cond - np.eye(length)) < 1e-5)

    def _tensor_kraus_operators(self, array1, array2):
        r"""Tensor kraus operators together needed for channel and entropy exchange computation."""
        output = kron(array1, array2)
        output.enable_caching()
        self._nth_kraus_ops_conj = output.conj().transpose((0, 2, 1))
        self._nth_kraus_ops_conj.enable_caching()
        return output

    def _kraus_operators_exchange(self, n):
        self._nth_kraus_exch = tensordot(self.nth_kraus_ops_conj, self.nth_kraus_ops,
                                         axes=((2), (1))).transpose(axes=(0, 2, 1, 3))

    def entropy_exchange(self, rho, n, adjoint=False):
        r"""
        Compute entropy exchange matrix for some rho.

        Parameters
        ----------
        rho : array
            Trace one, positive-semidefinite, hermitian matrix.

        n : int
            Integer of the number of times the channel is tensored.

        adjoint : bool
            Compute the adjoint of the complementary channel instead.

        Notes
        -----
        See Nielsen and Chaung book on the definition of entropy exchange.

        Returns
        -------
        list :
            Returns the complementary of the channel (ie trace one, positive-semidefinite,
            hermitian matrix) in sparse format.

        """
        if adjoint:
            rho_reshaped = rho.reshape((self.numb_kraus**n, self.numb_kraus**n, 1, 1))
            return np.sum(self.nth_kraus_exch * rho_reshaped, axis=(0, 1))
        w = matmul(self.nth_kraus_exch, rho)
        W = np.trace(w, axis1=2, axis2=3, dtype=np.complex128)
        return [W.T]

    def channel(self, rho, adjoint=False):
        r"""
        Compute the channel or it's adjoint on some density state rho.

        Parameters
        ----------
        rho : array
            Trace one, positive-semidefinite, hermitian matrix.

        adjoint : bool
            Compute the adjoint of the channel instead.

        Returns
        -------
        array :
            Compute the sparse channel A (or adjoint A^{\dagger}) based on the kraus operators on
            rho.

        """
        if adjoint:
            left_mult = matmul(self.nth_kraus_ops_conj, rho)
            return np.sum(matmul(left_mult, self.nth_kraus_ops), axis=0)
        return np.sum(matmul(matmul(self.nth_kraus_ops, rho), self.nth_kraus_ops_conj), axis=0)

    def average_entanglement_fidelity(self, probs, states):
        r"""
        Calculate average entanglement fidelity of an given ensemble.

        Equivalent to ensemble average fidelity.

        Parameters
        ----------
        probs : list
            List of probabilities for each state in the emsemble.

        states : list
            List of density states.

        Returns
        -------
        float :
            Average Entanglement fidelity of a ensemble.

        Notes
        -----
        Based on "arXiv:quant-ph/0004088".

        """
        avg_fid = 0.
        for i, p in enumerate(probs):
            avg_fid += p * np.sum(np.abs(np.trace(matmul(self.kraus_ops, states[i]), axis1=0))**2)
        return avg_fid

    @staticmethod
    def parallel_concatenate(kraus1, kraus2):
        r"""
        Parallel Concatenation of two kraus operators, A, B to produce "A tensored B".

        Parameters
        ----------
        kraus1 : list, array, SparseArray
            List or three-dimensional (numpy or sparse) array of kraus operators on left-side.

        kraus2: list, array, SparseArray
            List or three-dimensional (numpy or sparse) array of kraus operators on right-side.

        Returns
        -------
        array
            Three-dimensional array of kraus operators.

        Notes
        -----
        See Mark Wilde's book, "Quantum Information Theory"

        """
        assert isinstance(kraus1, (list, np.ndarray, SparseArray)), \
            "Kraus1 should be list or numpy array or sparse array."
        assert isinstance(kraus2, (list, np.ndarray, SparseArray)), \
            "Kraus2 should be list or numpy array or sparse array."
        # If they aren't in sparse format.
        if isinstance(kraus1, (list, np.ndarray)):
            kraus1 = COO(np.array(kraus1, dtype=np.complex128), )
        if isinstance(kraus2, (list, np.ndarray)):
            kraus2 = COO(np.array(kraus2, dtype=np.complex128), )
        return kron(kraus1, kraus2)

    @staticmethod
    def serial_concatenate(kraus1, kraus2):
        r"""
        Return Serial Concatenation of two kraus operators, A, B to produce "A composed B".

        Parameters
        ----------
        kraus1 : array or SparseKraus
            Three-dimensional array of kraus operators on left-side.

        kraus2: array or SparseKraus
            Three-dimensional array of kraus operators on right-side.

        Returns
        -------
        array
            Three-dimensional array of kraus operators.

        Notes
        -----
        See Mark Wilde's book, "Quantum Information Theory"

        """
        assert isinstance(kraus1, (list, np.ndarray, SparseArray)), \
            "Kraus1 should be list or numpy array or sparse array."
        assert isinstance(kraus2, (list, np.ndarray, SparseArray)), \
            "Kraus2 should be list or numpy array or sparse array."
        # If they aren't in sparse
        if isinstance(kraus1, (list, np.ndarray)):
            kraus1 = COO(np.array(kraus1, dtype=np.complex128),)
        if isinstance(kraus2, (list, np.ndarray)):
            kraus2 = COO(np.array(kraus2, dtype=np.complex128),)

        # Test that the dimensions of kraus operators match.
        for k2 in kraus2:
            for k1 in kraus1:
                if not k2.shape[0] == k1.shape[1]:
                    raise TypeError("Dimension of Kraus Operators should match each other.")

        # Multiply all kraus operators with one another
        kraus_ops = [matmul(kraus1[j], kraus2[i]) for j in range(0, len(kraus1))
                     for i in range(0, len(kraus2))]
        # Reshape them to be three-dimensional.
        return COO(np.array(kraus_ops, dtype=np.complex128),)


class DenseKraus(Kraus):
    r"""
    Kraus operators represented as Dense matrices.

    Attributes
    ----------
    kraus_ops : array
        Three-dimensional numpy array holding the initial (n=1) kraus operators provided by user.
    current_n : int
        How many times the channel A is tensored A^{\otimes n}.
    nth_kraus_ops : array
        Three-dimensional numpy array holding kraus operators represnting the channel tensored
        "current_n" times.

    Methods
    -------
    is_trace_perserving :
        Return true if the kraus operators represent a trace-perserving map.
    update_kraus_operators :
        Update the kraus operators to correspond to the channel tensored n times.
    channel(rho) :
        Apply the channel to a hermitian matrix rho.
    entropy_exchange(rho) :
        Apply the complementary channel to a hermitian matrix rho.
    average_entanglement_fidelity :
        Calculate the average entanglement fidelity with respect to an ensemble of states.
    parallel_concatenate :
        Given two channels A and B, return kraus operators representing "A tensored B".
    serial_concatenate :
        Given two channels A and B, return kraus operators representing "A composed B".

    Examples
    ---------
    # Construct Bit-Flip Map and apply to a channel.
    X = np.array([[0., 1.], [1., 0.]])
    I = np.array([[1., 0.], [0., 1.]])
    prob_error = 0.25
    kraus_ops = [np.sqrt(1 - prob_error) * I, np.sqrt(prob_error) * X]
    numb_qubits, dim_in, dim_out = [1, 1], 2, 2
    kraus = DenseKraus(kraus_ops, numb_qubits, dim_in, dim_out)
    rho = (I + 0.2 * X) / 2.
    channel_rho = DenseKraus.channel(rho, n=1)

    # Dephrasure Channel (Orthogonal Kraus Example)
    numb_qubits, dim_in = [1, 1], 2
    dim_out = 3
    Z = np.array([[1., 0.], [0., -1.], [0., 0.]])  # Kraus operator for dephasing channel.
    e1 = np.array([[0., 0.], [0., 0.], [0., 1.]])  # Kraus operators for erasure channel.
    e2 = np.array([[0., 0.], [0., 0.], [1., 0.]])  # Kraus operators for erasure channel.
    p_dep = 0.25  # Probability dephasing error
    p_era = 0.5  # Probability of erasure error
    kraus_ops = [np.sqrt(1 - p_era) * np.sqrt(1 - p_dep) * I,
                 np.sqrt(1 - p_era) * np.sqrt(p_dep) * Z,
                 np.sqrt(p_era) * e1, np.sqrt(p_era) * e2]
    ortho_kraus = [2]  #  Index of Erasure kraus in "kraus_ops" are orthogonal to dephasing ops.
    kraus = DenseKraus(kraus_ops, numb_qubits, dim_in, dim_out, orthogonal_kraus)

    # Apply to dephrasure channel tensored two times on a diagonal two-qubit density matrix.
    rho = np.diag([0.25, 0.25, 0.25, 0.25])
    channel = kraus.channel(rho, n=2)

    """

    def __init__(self, kraus_ops, numb_qubits, dim_in=2, dim_out=2, orthogonal_kraus=()):
        r"""
        Construct for the kraus operators represented as dense matrices.

        Parameters
        ----------
        kraus_ops : list, array
            List or numpy array containing all the kraus operators representing a quantum channel.
        numb_qubits : list
            Number of qubits [M, N] of input and output of the quantum channel.
        dim_in : int
            Dimension of a single-particle/qubit hilbert space. For qubit, dimension is two by
            definition.
        dim_out : int
            Dimension of a single-particle/qubit hilbert space of the output. For qubit, dimension
            is two by definition.
        orthogonal_kraus : list
            Set of M kraus operators \{A_1, ..., A_M\}. The list gives the index where the each
            group are orthogonal to one another.

        """
        if not isinstance(kraus_ops, (list, np.ndarray)):
            raise TypeError("Kraus operators should be a list or numpy array.")
        # Store a copy of the single kraus operators.
        self.kraus_ops = kraus_ops
        if isinstance(kraus_ops, list):
            self.kraus_ops = np.array(kraus_ops, dtype=np.complex128)
        self._orthogonal_kraus = orthogonal_kraus
        super(DenseKraus, self).__init__(kraus_ops, numb_qubits, dim_in=dim_in, dim_out=dim_out)

    @property
    def orthogonal_kraus(self):
        return self._orthogonal_kraus

    def __add__(self, other):
        r"""Return kraus operators that are serially concatenate to one another."""
        # Assert that dimensions of kraus operators match each other.
        if not self.dim_in**self.numb_qubits[0] == other.dim_out**other.numb_qubits[1]:
            raise TypeError("Kraus operator dimension do not match.")
        numb_qubits = [self.numb_qubits[0], other.numb_qubits[1]]
        dim_in, dim_out = other.dim_in, self.dim_out
        return DenseKraus(DenseKraus.serial_concatenate(self.kraus_ops, other.kraus_ops),
                          numb_qubits, dim_in, dim_out)

    def __mul__(self, other):
        r"""Return kraus operators that are parallel concatenated with one another."""
        assert isinstance(other, DenseKraus), "Kraus object should be of type '.DenseKraus'."
        # Assert that dimensions of kraus operators match each other.
        # The number of qubits is reset to one.
        numb_qubits = [1, 1]
        dim_in, dim_out = other.dim_in * self.dim_in, other.dim_out * self.dim_out
        return DenseKraus(DenseKraus.parallel_concatenate(self.kraus_ops, other.kraus_ops),
                          numb_qubits, dim_in, dim_out)

    def is_trace_perserving(self):
        r"""Return true if kraus operators are trace-perserving."""
        trace_cond = np.einsum("ikj, ijm-> km", np.conj(np.transpose(self.kraus_ops, (0, 2, 1))),
                               self.kraus_ops)
        length = self.kraus_ops[0].shape[1]

        return np.all(np.abs(trace_cond - np.eye(length)) < 1e-5)

    def _tensor_kraus_operators(self, array1, array2):
        r"""Tensor Sets of kraus operators together, See Parallel Concantenation."""
        return np.kron(array1, array2)

    def _kraus_operators_exchange(self, n):
        r"""Help set up the kraus operators needed for entropy exchange function."""
        self._nth_kraus_exch = np.tensordot(np.conj(np.transpose(self._nth_kraus_ops, (0, 2, 1))),
                                            self._nth_kraus_ops, axes=((2), (1))).swapaxes(1, 2)

    def _initialize_kraus_ops(self):
        r"""Initialize the kraus operators for computation of channel and entropy exchange."""
        self._nth_kraus_ops = self.kraus_ops.copy()
        self._nth_kraus_exch = np.tensordot(np.conj(np.transpose(self._nth_kraus_ops, (0, 2, 1))),
                                            self._nth_kraus_ops, axes=((2), (1))).swapaxes(1, 2)

    def channel(self, rho, adjoint=False):
        r"""
        Compute the channel or it's adjoint on some density state rho.

        Parameters
        ----------
        rho : array
            Trace one, positive-semidefinite, hermitian matrix. If rho is two-dimensional, then
            channel is computed only on rho. If rho is three-dimensional with shape (M, d, d),
            where M is the number of density matrices. Then Channel is computed on each M density
            matrices.

        adjoint : bool
            Compute the adjoint of the channel instead.

        Returns
        -------
        array :
            Compute the channel A based on the kraus operators on rho. If rho is two-dimensional,
            it returns two-dimensional A(rho). If rho is three-dimensional, it returns a
            three-dimensional array [A(rho_1), .., A(rho_M)].

        """
        assert isinstance(rho, np.ndarray), "Density State rho should be numpy array."
        if adjoint:
            # Multiply left by hermitian tranpose kraus operators
            left_multi = np.conj(np.transpose(self.nth_kraus_ops, (0, 2, 1)))
            right_multi = self.nth_kraus_ops
        else:
            left_multi = self.nth_kraus_ops
            right_multi = np.conj(np.transpose(self.nth_kraus_ops, (0, 2, 1)))

        # If want to compute the channel across many density matrices or just one density matrix.
        if rho.ndim == 2:
            kraus_mul_rho = np.matmul(left_multi, rho)
            channel = np.matmul(kraus_mul_rho, right_multi)
            channel = np.einsum("ijk->jk", channel)
        else:
            kraus_mul_rho = np.einsum("ijk,lkp->lijp", left_multi, rho)
            channel = np.einsum("lijp, ipm->lijm", kraus_mul_rho, right_multi)
            channel = np.einsum("lijm->ljm", channel)
        return channel

    def entropy_exchange(self, rho, n, adjoint=False):
        r"""
        Compute complementary channel/entropy exchange matrix for some rho.

        Parameters
        ----------
        rho : array
            Trace one, positive-semidefinite, hermitian matrix.

        n : int
            Integer of the number of times the channel is tensored.

        adjoint : bool
            Compute the adjoint of the complementary channel instead. This doesn't use exploit
            orthogonal kraus operators.

        Notes
        -----
        If 'orthogonal_kraus' class attribute have been provided, then the kraus operators are
        broken down into M sets [A_1, .., A_M], where each element of the set A_i is orthogonal to
        each element of the set A_j (i not equal to j). As a result, the complementary channel
        becomes a block-diagonal matrix with each M blocks corresponding to block A_i.

        See Nielsen and Chaung book on the definition of entropy exchange.

        Returns
        -------
        list :
            Returns the complementary of the channel (ie trace one, positive-semidefinite,
            hermitian matrix). If 'orthogonal_kraus' class attribute have been provided,
            then it returns a list [A_1(rho), .., A_n(rho)] of the block-diagonal components
            (see Notes).

        Examples
        --------
        # Erasure example with orthogonal kraus sets
        k0 = np.array([[1, 0.], [0., 1.], [0., 0.]]) #  identity kraus operator
        k1 = np.array([[0., 0.], [0., 0.], [1., 0.]]) #  Erasure kraus operator 1
        k2 = np.array([[0., 0.], [0., 0.], [0., 1.]]) #  Erasure kraus operator 2
        # The set {k0} is orthogonal to the set {k1, k2}.
        orthogonal_kraus = [1]
        kraus = DenseKraus([k0, k1, k2], [1, 1], 2, 3)
        channel = kraus.entropy_exchange( some random density matrix rho)
        # Channel should be a list of two matrices [A_1, A_2], A_1 corresponds to {k_0}
        # and A_2 correponds to {k_1, k_2}.

        """
        numb_rows = self.numb_kraus ** n
        # Compute adjoint of the complementary channel.
        if adjoint:
            return np.einsum("ijkl,ij->kl", self.nth_kraus_exch, rho, dtype=np.complex128)

        if len(self.orthogonal_kraus) == 0:
            W = np.einsum("ijkl,lk->ij", self.nth_kraus_exch, rho, dtype=np.complex128)
            return [W.T]
        else:
            output = []
            count = 0
            for i in self.orthogonal_kraus:
                orthogonal_split = i * self.numb_kraus ** (n - 1)

                W = np.einsum("ijkl,lk->ij", self.nth_kraus_exch[count:orthogonal_split,
                              count:orthogonal_split, :, :], rho)
                output.append(W.T)
                count += orthogonal_split

            # Last section of kraus operators
            diff = (numb_rows - count)
            W = np.einsum("ijkl,lk->ij", self.nth_kraus_exch[diff:, diff:, :, :], rho)
            output.append(W.T)
        return output

    def average_entanglement_fidelity(self, probs, states):
        r"""
        Calculate average entanglement fidelity of an given ensemble.

        Equivalent to ensemble average fidelity.

        Parameters
        ----------
        probs : list
            List of probabilities for each state in the emsemble.

        states : list
            List of density states.

        Returns
        -------
        float :
            Average Entanglement fidelity of a ensemble.

        Notes
        -----
        Based on "arXiv:quant-ph/0004088".

        """
        avg_fid = 0.
        for i, p in enumerate(probs):
            avg_fid += p * np.sum(np.abs(np.einsum("ijk,kp->i", self.kraus_ops, states[i]))**2)
        return avg_fid

    @staticmethod
    def parallel_concatenate(kraus1, kraus2):
        r"""
        Parallel Concatenation of two kraus operators, A, B to produce "A tensored B".

        Parameters
        ----------
        kraus1 : array, list
            Three-dimensional array or list of kraus operators on left-side.

        kraus2: array, list
            Three-dimensional array or list of kraus operators on right-side.

        Returns
        -------
        array
            Three-dimensional array of kraus operators.

        Notes
        -----
        See Mark Wilde's book, "Quantum Information Theory"

        Examples
        --------
        # Define I, X, Y, Z to be pauli channels
        k1 = [I, X]  # As a list
        k2 = np.array([I, Z])  # As numpy array
        k3 = DenseKraus.parallel_concatenate(k1, k2)  # Parallel Concatenate two kraus op sets.
        # k3 should be numpy array of [I tensor I, I tensor Z, X tensor I, X tensor Z]

        """
        # Convert list type to numpy array.
        if isinstance(kraus1, list):
            kraus1 = np.array(kraus1, dtype=np.complex128)
        if isinstance(kraus2, list):
            kraus2 = np.array(kraus2, dtype=np.complex128)
        # Assert that kraus operators be three-dimensional.
        if isinstance(kraus1, np.ndarray):
            assert kraus1.ndim == 3, "Set of kraus operators should be three-dimensional np array."
        if isinstance(kraus2, np.ndarray):
            assert kraus2.ndim == 3, "Set of kraus operators should be three-dimensional np array."
        return np.kron(kraus1, kraus2)

    @staticmethod
    def serial_concatenate(kraus1, kraus2):
        r"""
        Seriel Concatenation of two kraus operators, A, B to produce "A composed B".

        Parameters
        ----------
        kraus1 : list, array
            Three-dimensional array or list of kraus operators A on left-side.

        kraus2: list, array
            Three-dimensional array or list of kraus operators B on right-side.

        Returns
        -------
        array
            Three-dimensional array of kraus operators.

        Notes
        -----
        See Mark Wilde's book, "Quantum Information Theory"

        Examples
        --------
        # Define I, X, Y, Z to be pauli channels
        k1 = [I, X]  # As a list
        k2 = np.array([I, Z])  # As numpy array
        k3 = DenseKraus.serial_concatenate(k1, k2)  # Serial Concatenate of two kraus op sets.
        # k3 should be numpy array of [I, Z, X, X.dot(Z)]

        """
        assert isinstance(kraus1, (list, np.ndarray)), "Kraus1 should be list or numpy array."
        assert isinstance(kraus2, (list, np.ndarray)), "Kraus2 should be list or numpy array."
        # Convert list type to numpy array.
        if isinstance(kraus1, list):
            kraus1 = np.array(kraus1, dtype=np.complex128)
        if isinstance(kraus2, list):
            kraus2 = np.array(kraus2, dtype=np.complex128)
        # Assert that kraus operators be three-dimensional.
        if isinstance(kraus1, np.ndarray):
            assert kraus1.ndim == 3, "Set of kraus operators should be three-dimensional np array."
        if isinstance(kraus2, np.ndarray):
            assert kraus2.ndim == 3, "Set of kraus operators should be three-dimensional np array."

        # Test that the dimensions of kraus operators match.
        for k2 in kraus2:
            for k1 in kraus1:
                if not k2.shape[0] == k1.shape[1]:
                    raise TypeError("Dimension of Kraus Operators should match each other.")

        # Multiply all kraus operators with one another
        kraus_ops = np.einsum("ijk,mkl->imjl", kraus1, kraus2)
        x1, x2, y, z = kraus_ops.shape
        # Reshape them to be three-dimensional.
        return np.reshape(kraus_ops, (x1 * x2, y, z))
