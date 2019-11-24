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
from scipy.linalg import sqrtm

from qutip import Qobj, ptrace


__all__ = ["ChoiQutip"]

"""
Choi operator class based on QuTIP package.
Used for modelling recovery and encoders for optimized-based codes.
"""


class ChoiQutip():
    r"""
    Choi Operator (I \otimes \Phi) of quantum channel \Phi.

    This class does not support doing parallel or serial concatenation of channels as choi matrices.

    Attributes
    ----------
    choi : array
        Choi-Matrix.
    num_in : int
        Number of input particles/qubits of the channel \Phi.
    num_out : int
        Number of output particles/qubits of the channel \Phi.
    input_dim : int
        Dimension of input hilbert space of all particles of the channel \Phi.
    output_dim : int
        Dimension of output hilbert space of all particles of the channel \Phi.
    qutip : Qobj
        The qutip object for the superoperator representing the choi-matrix.

    Methods
    -------
    channel :
        Calculate the channel of a given density matrix.
    complementary_channel :
        Calculate the complementary channel of a given density matrix.
    average_entanglement_fidelity :
        Comute the average entanglement fidelity of a given ensemble.
    kraus_operators :
        Get the kraus operators from the choi-matrix.
    one_qubit_choi_matrix :
        Given six-parameters, return the choi-matrix correponding to a one-qubit channel.

    Notes
    -----
    - With this definition of choi-matrix, then column-stacking vectorization has to be used,
        ie [[a, b], [c, d]]-> [a, c, b, d]. QuTip defines choi-matrix similarly. John Watrous
        defines in his book as (\Phi \otimes I) and uses row-stacking vectorization. The function
        "_vectorize" is the vectorization used in this class.


    Examples
    --------
    # Erasure Channel
    >> choi = "Choi matrix of erasure channel"
    >> numb_qubits = [1, 1]
    >> dim_in = 2
    >> dim_out = 3 # Third dimension corresponds to erased space.

    # Dephrasure channel
    >> choi = "Insert Choi matrix of dephrasure channel"
    >> numb_qubits = [1, 1] # One qubit maps to one qubit
    >> dim_in = 2 # Dimension of a single qubit
    >> dim_out = 3 # Third dimension corresponds to erasure basis.

    """
    def __init__(self, choi_matrix, numb_qubits, dim_in, dim_out):
        r"""
        The Choi matrix should be defined as $(I \otimes \Phi)$.

        Parameters
        ----------
        choi_matrix : array
            The choi matrix representing the quantum channel. Should be trace-perserving and
            completely positive.

        numb_qubits : list
            Should be [N, M], where N is number of qubits of the input of the channel, and M is
            the expected number of qubits of the output of the channel.

        dim_in : int
            The dimension of a single-qubit hilbert space of the input. Generally is two.

        dim_out : int
            The dimension of a single-qubit hilbert space of the output. Generally is two,
            sometimes three.

        Raises
        ------
        AssertionError :
            Raises an error if the choi matrix provided is not trace-perserving and completely
            positive.

        ValueError :
            If the dimensions from 'numb_qubits', 'dim_in' and 'dim_out' do not match the shape
            of the choi matrix.

        """
        self.choi = choi_matrix
        self._num_in = numb_qubits[0]
        self._num_out = numb_qubits[1]
        # Dimension of input/output Hilbert space.
        self._dim_in = dim_in
        self._dim_out = dim_out
        self.input_dim = dim_in**numb_qubits[0]
        self.output_dim = dim_out**numb_qubits[1]

        # Check choi matrix matches the dimensions
        if self.choi.shape[1] != self.output_dim * self.input_dim:
            raise ValueError("Choi matrix shape does not match dimension specified.")

        # dimemsion for the qutip object to understand.
        dim = [[[dim_in] * self.num_in, [dim_out] * self.num_out],
               [[dim_in] * self.num_in, [dim_out] * self.num_out]]
        self._dim = dim
        self.qutip = Qobj(choi_matrix, dims=dim, type="super", superrep="choi")
        assert self.qutip.istp, "Choi matrix is not trace perserving."
        assert self.qutip.iscp, "Choi matrix is not completely positive."

    @property
    def dim_in(self):
        return self._dim_in

    @property
    def dim_out(self):
        return self._dim_out

    @property
    def dim(self):
        return self._dim

    @property
    def num_in(self):
        return self._num_in

    @property
    def num_out(self):
        return self._num_out

    @property
    def numb_qubits(self):
        return [self.num_in, self.num_out]

    def channel(self, rho):
        r"""
        Compute the channel on a density matrix.

        Parameters
        ----------
        rho : array

        Returns
        -------
        array :

        References
        ----------
        See John Watrous Notes of Choi-Matrix. Note that he uses row-based vectorization.
        """
        # Definition is taken from John Watrous.
        choi_rho = self.qutip.data.dot(np.kron(rho.T, np.eye(self.output_dim)))

        # Keep the following dimensions when doing partial trace
        keep = [i for i in range(self.num_out, 2 * self.num_out)]
        return ptrace(Qobj(choi_rho, dims=self.dim), keep).data.todense()

    def complementary_channel(self, rho):
        r"""
        Compute the channel (I \otimes \Phi)(\rho) of a density matrix \rho.

        Used in alternative method of calculating coherent information, explicity this can be
        used rather than using the entropy exchange matrix (defined in Nielsen and Chaung).

        Parameters
        ----------
        rho : array

        Returns
        -------
        array :

        References
        ----------
        - Liu, Tong. "Parameter Regime Giving Zero Quantum Coherent Information of A Non-Pauli
            Quantum Channel." PhD diss., 2015.
        """
        # Follows from Tong's Thesis, Bei's Student.
        square_root = sqrtm(rho)
        tensored = np.kron(square_root.T, np.eye(self.output_dim))
        return tensored.dot(self.choi).dot(tensored)

    def _vectorize_rho(self, rho):
        # Vectorize rho using column-stacking.
        return np.ravel(rho, "F")

    def average_entanglement_fidelity(self, probs, states):
        r"""
        Average entanglement fidelity of a given ensemble of density states.

        For the purposes of channel-adaptive codes, the ensemble should be maximally mixed state
        with probability one. This function only works when choi matrix maps qubits to qubits.

        Parameters
        ----------
        probs : list
            List of probabilities for each state in the ensemble.

        states : list
            List of density states.

        Returns
        -------
        float :
            Average entanglement fidelity.

        Notes
        -----
        See "doi:10.1017/CBO9781139034807" Chapter 13.

        """
        if self.input_dim != self.output_dim:
            raise ValueError("Channel must map two-dimensional m-qubits to two-dimensional "
                             "n-qubits.")
        avg_fid = 0.
        for i, p in enumerate(probs):
            state_vec = self._vectorize_rho(states[i])
            avg_fid += p * np.conj(state_vec.T).dot(self.choi).dot(state_vec)
        return avg_fid

    def kraus_operators(self):
        r"""
        Obtain kraus operators from choi matrix.

        Returns
        -------
        list
            list of 2-dimensional numpy arrays each correspond to kraus operators.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.choi)
        krauss_ops = []
        for i in range(0, eigenvectors.shape[0]):
            if eigenvalues[i] > 1e-8:
                eigenvec = eigenvectors[:, i]
                krauss_shape = (self.output_dim, self.input_dim)
                output = eigenvec.reshape(krauss_shape, order="F")
                krauss_ops.append(output * np.sqrt(eigenvalues[i]))
        return krauss_ops

    @staticmethod
    def constraint_partial_trace(numb_qubits, dim_in, dim_out):
        r"""
        Conditions for 'picos.tools.partial_trace to satisfy trace-perserving for choi matrices.

        The choi matrix C is defined to be matrix :math:'(I \otimes \Phi)(\sum_{ij} \ketbra{i}{j})',
        of a quantum channel :math:'\Phi' and over all basis-vectors :math:'\ketbra{i}{j}'.
        The trace-perserving conditions implies that the trace over the subsystem H is the identity.
        In other words, :math:'Tr_{H}(C) == I_{dim(H)}'.

        Parameters
        ----------
        numb_qubits : list
            Should be [N, M], where N is number of qubits of the input of the channel, and M is
            the expected number of qubits of the output of the channel.

        dim_in : int
            The dimension of a single-qubit hilbert space of the input. Generally is two.

        dim_out : int
            The dimension of a single-qubit hilbert space of the output. Generally is two,
            sometimes three.

        Returns
        -------
        (int, list):
            Integer corresponds to which index to trace out, by default is one. The list indicates
            the matrix dimensions of A then B of respectively (A \otimes B).
        """
        # Dimensions of identity channel I and channel \Phi of (I \otimes \Phi), respectively.
        dims = [(dim_in ** numb_qubits[0], dim_in ** numb_qubits[0]),
                (dim_out ** numb_qubits[1], dim_out ** numb_qubits[1])]
        # The matrix to partial trace over is the $\Phi$
        k = 1
        return k, dims
