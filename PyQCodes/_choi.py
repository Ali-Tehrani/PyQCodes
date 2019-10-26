import numpy as np
from scipy.linalg import sqrtm
import picos as pic  # Semi-definite Programming

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
    choi = "Choi matrix of erasure channel"
    numb_qubits = [1, 1]
    dim_in = 2
    dim_out = 3 # Third dimension corresponds to erased space.

    # Dephrasure channel
    choi = "Insert Choi matrix of dephrasure channel"
    numb_qubits = [1, 1] # One qubit maps to one qubit
    dim_in = 2 # Dimension of a single qubit
    dim_out = 3 # Third dimension corresponds to erasure basis.

    # TODO: Why did I write this.
    In addition, I'm assuming both the underlying input and output hilbert space is qubit and
    that the choi-matrix and density matrices are both written in the same dimension.
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
        self.num_in = numb_qubits[0]
        self.num_out = numb_qubits[1]
        # Dimension of input/output Hilbert space.
        self.dim_out = dim_out
        self.input_dim = dim_in**numb_qubits[0]
        self.output_dim = dim_out**numb_qubits[1]

        # Check choi matrix matches the dimensions
        if self.choi.shape[1] != self.output_dim * self.input_dim:
            raise ValueError("Choi matrix shape does not match dimension specified.")

        # dimemsion for the qutip object to understand.
        dim = [[[dim_in] * self.num_in, [dim_out] * self.num_out], \
               [[dim_in] * self.num_in, [dim_out] * self.num_out]]
        self.dim = dim
        self.qutip = Qobj(choi_matrix, dims=dim, type="super", superrep="choi")
        assert self.qutip.istp, "Choi matrix is not trace perserving."
        assert self.qutip.iscp, "Choi matrix is not completely positive."

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

    def average_entanglement_fidelity(self, probs, states, krauss=None):
        r"""
        Average entanglement fidelity of a given ensemble of density states.

        For the purposes of channel-adaptive codes, the ensemble should
        be maximally mixed state with probability one. This function only
        works when choi matrix maps qubits to qubits.

        Parameters
        ----------
        probs : list
            List of probabilities for each state in the ensemble.

        states : list
            List of density states.

        krauss : array
            Multi-dimensional array of krauss operators. Used for optimizing
            the average entanglement fidelity of a ensemble.

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

        # Used for channel-adaptive codes.
        if krauss is not None:
            matrix = np.zeros((krauss[0].size, krauss[0].size), dtype=np.complex128)
            kraus_conj = np.conj(np.transpose(krauss, (0, 2, 1)))
            for i, p in enumerate(probs):
                for j, k in enumerate(krauss):
                    matrix += p * np.outer(states[i].dot(kraus_conj[j]), states[i].dot(k))
            return np.trace(self.choi.dot(matrix))

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
    def one_qubit_choi_matrix(l_vec, t_vec, pauli_errors=False):
        r"""
        Choi matrix based on the one qubit parameterization.

        Parameters
        ----------
        l_vec : array
            [lx, ly, lz], the lambdas parameters. If pauli_errors is true, then l_vec denotes the
            set of pauli errors p_x, p_y, p_z, where p_x denotes the probability of bit-flip X
            error, and p_y, p_z denotes the probability of Y and phase-flip error Z, respectively.

        t_vec : array
            [Tx, Ty, TZ] the non-unital parameters, each correponding to a translation of the
            maximally mixed state by the quantum channel.

        pauli_errors : bool
            True if l_vec is the set of pauli errors ie l_vec = [px, py, pz], where px is the
            probability of pauli-X error, etc.

        Returns
        -------
        ChoiQutip :
            Returns ChoiQuTip based on the choi matrix of the one-qubit channel.

        Raises
        ------
        AssertionError :
            If the pauli errors [px, py, pz] are not all less than or equal to one.

        ValueError :
            If the six parameters [lx, ly, lz, tx, ty, tz] do not satisfy the conditions of being
            completely positive and trace-perserving qubit channel.

        Notes
        -----
        Based on the paper "doi:10.1088/1751-8113/47/13/135302".
        """
        def _convert_pauli_errors_to_six_parameters(l_vec):
            px, py, pz = l_vec
            p0 = (1 - px - py - pz)
            lx = (p0 + px - py - pz)
            ly = (p0 - px + py - pz)
            lz = (p0 - px - py + pz)
            return lx, ly, lz

        def _conditions_non_unital(lamdas, t_1, t_2, t_3):
            # Conditions for Completely Positive and Trace-Perserving For Non-Unital Qubit Channels.
            t = np.array([t_1, t_2, t_3])
            normt = np.linalg.norm(t)
            u = t / normt
            q0 = (1 + np.sum(lamdas)) / 4.
            q1 = (1 + np.sum(lamdas * np.array([1, -1, -1]))) / 4.
            q2 = (1 + np.sum(lamdas * np.array([-1, 1, -1]))) / 4.
            q3 = (1 + np.sum(lamdas * np.array([-1, -1, 1]))) / 4.
            q = 256 * q0 * q1 * q2 * q3
            r = 1. - np.sum(lamdas ** 2) + 2. * np.sum(lamdas ** 2 * u ** 2)
            return normt ** 2., r - np.sqrt(r ** 2 - q), np.array([q0, q1, q2, q3])

        # Initialize the parameters.
        if pauli_errors:
            assert np.all(l_vec) <= 1., "Pauli errors [px, py, pz] have to be less(equal) to one."
            lx, ly, lz = _convert_pauli_errors_to_six_parameters(l_vec)
        else:
            lx, ly, lz = l_vec
        tx, ty, tz = t_vec

        # Test if Conditions are met to satisfy completely positive and trace-perserving.
        if np.all(np.abs(t_vec) < 1e-5):
            # Unital Channels
            assert -1e-10 >= np.abs(lx + ly) - (1 + lz)
            assert -1e-10 >= np.abs(lx - ly) - (1 - lz)
        else:
            # Non-unital channels
            cond1, cond2, cond3 = _conditions_non_unital(l_vec, tx, ty, tz)
            satisfy_cp_tp_condi = -1e-10 <= cond2 - cond1 and -cond2 - cond1 <= 1e-10 and \
                                  np.all(cond3 >= -1e-8)
            if not satisfy_cp_tp_condi:
                raise ValueError("Lambda and t values do not satisfy CP and TP conditions.")

        # Construct the one-qubit choi-matrix from paper.
        choi = 0.5 * np.array([[1 + tz + lz, complex(tx, -ty), 0., lx + ly],
                               [complex(tx, ty), 1 - tz - lz, lx - ly, 0.],
                               [0., lx - ly, 1 + tz - lz, complex(tx, -ty)],
                               [lx + ly, 0., complex(tx, ty), 1 - tz + lz]], dtype=np.complex128)
        return ChoiQutip(choi, [1, 1], 2, 2)

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
