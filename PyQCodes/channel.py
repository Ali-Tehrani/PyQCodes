from PyQCodes._kraus import SparseKraus, DenseKraus
from PyQCodes._choi import ChoiQutip
from PyQCodes._coherent import optimize_procedure

import numpy as np
from scipy.linalg import sqrtm  # For Fidelity.
from abc import ABC, abstractmethod

# TODO: ADD Module Level-Documentation Here for AnalyticQCodes.
r"""


Contains AnalyticQChan which models quantum channels as krauss operators 
or choi matrices, and QDeviceChannel which models quantum channels as 
quantum circuits.
"""

__all__ = ["AnalyticQChan", "QDeviceChannel"]


class QChanABC(ABC):
    r"""Abstract Base Class for Quantum Channels."""
    def __init__(self, dim):
        pass

    @abstractmethod
    def coherent_info(self, rho):
        pass

    @abstractmethod
    def effective_channel(self, n):
        pass

    @abstractmethod
    def fidelity(self):
        pass

    @abstractmethod
    def statistics(self):
        pass

    @abstractmethod
    def channel(self):
        pass


class AnalyticQChan():
    r"""
    Class of Analytic Quantum Channel represented as either Kraus Operators or a Choi Matrix.

    Attributes
    ----------
    kraus : DenseKraus or SparseKraus
        Kraus operators of the channel.
    choi : ChoiQutip
        The choi object based on the class ChoiQuTip. This attribute is only available if the
        choi matrix is provided as 'operator' instance.
    numb_krauss :  int
        The number of kraus operators. Only available if kraus operators were provided as
        'operator' instance.
    kraus_ops : list
        Returns the kraus operators.

    Methods
    -------
    fidelity_two_states :
        The fidelity between two density states.
    average_entanglement_fidelity :
        The average entanglement fidelity with respect to a ensemble of states.
    channel :
        Calculates the output of the channel with respect to a density matrix.
    entropy_exchange :
        Calculates the entropy exchange matrix with respect to a density matrix.
    entropy :
        Calculates the entropy with respect to a density matrix.
    coherent_information :
        Calculates the coherent information with respect to a density matrix.
    optimize_coherent :
        Maximizes the coherent information of a given channel.
    optimize_fidelity :
        Minimizes the fidelity between pure-states of a given channel.

    Notes
    -----
    - Although, this class accepts both kraus operators and choi-matrices. It is highly recommended
    for computational speed-ups to use kraus operators rather than choi-matrices. Furthermore,
    with using choi-matrices, this class cannot compute the channel tensored with itself n-times,
    whereas with kraus operators it can be modeled as such.

    - It is recommended for large n or large number of kraus operators, to use the sparse option
    for kraus operators.

    References
    ----------
    For information on coherent information, see Mark Wilde's book, "Quantum Information Theory".
    For information on average entanglement fidelity, see Lidar's book, "Quantum Error-Correction".

    Examples
    --------
    # Example to compute coherent information of bit-flip map tensored two times.

    numb_qubits = [1, 1]  # Accepts one qubits and outputs one qubits.
    dim_in = 2  # Dimension of single qubit hilbert space is 2.
    dim_out = 2  # Dimension of single qubit hilbert space after the bit-flip channel is 2.

    p = 0.25  # Probability of error
    identity = np.sqrt(p) * np.eye(2)
    bit_flip = np.sqrt(1 - p) * np.array([[0., 1.], [1., 0.]])
    kraus_ops = [identity, bit_flip]
    channel = AnalyticQChan(kraus_ops, numb_qubits, dim_in, dim_out)

    # Optimize the coherent information using "differential evolution" and parameterization
    # of density matrix with OverParameterization.
    res = channel.optimize_coherent(n=2, rank=4, optimizer="diffev", param="overparam",
                                    maxiter=200)
    print(res)
    """

    def __init__(self, operator, numb_qubits, dim_in, dim_out, orthogonal_krauss=(), sparse=False):
        r"""
        Constructor of AnalyticQChan class.

        Parameters
        ----------
        operator : list or np.array
            If provided as a list, then it is assumed that each element of the list are the kraus
            operators for the channel. If "np.array", then it is assumed to be a two-dimensional
            array representing the choi matrix of the channel.
        numb_qubits : tuple
            Tuple (M, N) representing the number of particles/qubits M that are the input of the
            channel and number of particles/qubits N that are the output of the channel.
        dim_in : int
            The dimension of a single hilbert space of the input of the channel.  Normally,
            dealing with qubit input so dimension is two.
        dim_out : int
            The dimension of a single hilbert space of the output of the channel. Normally,
            dealing with qubit output so dimension is two.
        orthogonal_krauss : tuple
            Gives the indices, where the sets of kraus operators are orthogonal with one another
            ie A * B = 0, if A and B are within the same set.
        sparse : bool
            If true, then the kraus operators are modeled as sparse matrices. Cannot be used if
            operator is choi matrix.

        Raises
        ------
        TypeError :
            If operator is not a list or numpy array. The dimensions (dim_in, dim_out,
            numb_qubits) provided must match the shape of the operators.
        AssertionError :
            If operator is numpy array, then it must be a two-dimensional numpy array
            representing a matrix.

        Examples
        --------
        # Constructing a simple qubit channel
        numb_qubits = [1, 1]  # Accepts one qubits and outputs one qubits.
        dim_in = 2  # Dimension of single qubit hilbert space is 2.
        dim_out = 2  # Dimension of single qubit hilbert space after the qubit channel is 2.

        p = 0.25  # Probability of error
        identity = np.sqrt(p) * np.eye(2)
        bit_flip = np.sqrt(1 - p) * np.array([[0., 1.], [1., 0.]])
        kraus_ops = [identity, bit_flip]
        channel = AnalyticQChan(kraus_ops, numb_qubits, dim_in, dim_out, sparse=True)


        # Constructing the erasure channel where dimension of output hilbert space is larger.
        numb_qubits = [1, 1]
        dim_in = 2
        dim_out = 3  # Erasure channel adds a extra dimension to output of the channel.

        p = 0.25  # Probability of no error
        identity = np.sqrt(p) * np.array([[1., 0.], [0., 1.], [0., 0.]])
        e1 = np.sqrt(1 - p) * np.array([[0., 0.], [0., 0.], [1., 0.]])
        e2 = np.sqrt(1 - p) * np.array([[0., 0.], [0., 0.], [0., 1.]])

        kraus_ops = [identity, e1, e2]
        # The indices {0} of kraus ops is orthogonal to kraus ops indices to {1, 2}
        orthogonal = [1]
        channel = AnalyticQChan(kraus_ops, numb_qubits, dim_in, dim_out,
                                orthogonal_krauss=orthogonal)
        """
        if not (isinstance(operator, list) or isinstance(operator, np.ndarray)):
            # Test if it is kraus operator or choi type.
            raise TypeError("Operator argument should be a list (kraus) or ndarray (choi).")

        # If Operator is Kraus Type
        self.sparse = sparse
        if isinstance(operator, list):
            self._type = "kraus"
            if sparse:
                self.krauss = SparseKraus(operator, numb_qubits, dim_in, dim_out)
            else:
                self.krauss = DenseKraus(operator, numb_qubits, dim_in, dim_out, orthogonal_krauss)
            assert self.krauss.is_trace_perserving(), "Kraus operators provided are not " \
                                                      "trace-perserving."

        # If Operator is Choi Type
        elif isinstance(operator, np.ndarray):
            # Must be completely-positive and trace-perserving.
            assert operator.ndim == 2, "Choi matrix is two-dimensional array. Instead it is %d." \
                                       % operator.ndim
            self._type = "choi"
            self.choi = ChoiQutip(operator, numb_qubits, dim_in, dim_out)
        # TODO: Add or remove abstract class? Should be thought of after.
        # super(QChanABC, self).__init__(numb_qubits)

    @property
    def nth_kraus_operators(self):
        # Get kraus operators for the nth-channel use.
        assert self._type == "kraus", "Function only works with kraus operators."
        return self.krauss.nth_kraus_ops

    @property
    def input_dimension(self):
        # Get dimension of the single particle hilbert space tensored "numb_qubits" times.
        # Should be number of rows/columns of density state, that is about to go through channel.
        if self._type == "kraus": return self.krauss.input_dim**self._current_n
        else: return self.choi.input_dim

    @property
    def output_dimension(self):
        # Get dimension of density states of output of the channel based on current n.
        # Should be hte number of rows/columns of density state after going through the channel.
        if self._type == "kraus": return self.krauss.output_dim**self._current_n
        else: return self.choi.output_dim

    @property
    def _current_n(self):
        return self.krauss.current_n

    @property
    def numb_krauss(self):
        assert self._type == "kraus", "Function only works with kraus operators."
        return self.krauss.numb_krauss

    @property
    def kraus_ops(self):
        # Kraus operators for n=1 channel.
        if self._type == "kraus":
            return self.krauss.kraus_ops
        else:
            return self.choi.kraus_operators()

    def __add__(self, other):
        return AnalyticQChan(self.number_qubits, self.krauss + other.krauss)

    def __mul__(self, other):
        return AnalyticQChan(self.number_qubits + other.number_qubits,
                             self.krauss * other.krauss)

    def _is_qubit_channel(self):
        r"""True if it is qubit channel."""
        if self.input_dimension == 2 and self.output_dimension == 2:
            return True
        return False

    def fidelity_two_states(self, rho):
        # TODO: Add assertion that it must match dimensions.
        chan = self.channel(rho, n=1)
        sqrt_rho = sqrtm(rho)
        fidel = np.trace(sqrtm(sqrt_rho.dot(chan).dot(sqrt_rho)))
        assert np.abs(np.imag(fidel)) < 1e-5
        return np.real(fidel)

    def average_entanglement_fidelity(self, probs, states):
        r"""
        Calculates average entanglement fidelity of an given ensemble.

        Equivalent to ensemble average fidelity.

        Parameters
        ----------
        probs : list
            List of probabilities for each state in the ensemble.

        states : list
            List of density states.

        Returns
        -------
        float :
            Average Entanglement fidelity of a ensemble.

        References
        ----------
        Based on "arXiv:quant-ph/0004088".
        """
        if self._type == "kraus":
            return self.krauss.average_entanglement_fidelity(probs, states)
        return self.choi.average_entanglement_fidelity(probs, states)

    def channel(self, rho, n, adjoint=False):
        r"""
        Output of channel of a density matrix.

        Parameters
        ----------
        rho : array(2-dimensional array or Nth-dimensional Array)
            rho is either a single matrix acting as input or an array of single matrices
            acting as input to the quantum channel.

        n : integer
            Nth use of the channel.

        adjoint : bool
            True then adjoint of channel acting on rho is returned.

        Returns
        -------
        array: 2-dimensional or n-dimensional
            Output of the channel, either a single matrix or a list of single matrices.

        Notes
        -----
        It is highly recommended to use kraus operators rather than choi-matrices for
        computational speed-ups.
        """
        # For Kraus Operators
        if self._type == "kraus":
            # Check if krauss operator corresponds to the channel tensored n times.
            if n != self._current_n:
                self.krauss.update_kraus_operators(n)
            return self.krauss.channel(rho, adjoint)

        # For Choi Matrix.
        assert n == 1, "n, nth use of the channel must be one for choi-matrices."
        return self.choi.channel(rho)

    def entropy_exchange(self, rho, n, adjoint=False):
        r"""
        Computes the entropy exchange matrix given kraus operators.

        The entropy exchange matrix W for a set of kraus operators :math:'\{A_i\}' is defined
        to have matrix entries :math:'W_{ij}' on the ith, jth position to be

        .. math:: W_{ij} = (Tr(A_i \rho A_j^\dagger)).

        Parameters
        ----------
        rho : array
            Trace one, positive-semidefinte, hermitian matrix.

        n : int
            Compute entropy exchange for the channel tensored n times.

        adjoint : bool
            True if the adjoint of the complementary channel is computed.

        Returns
        -------
        list of arrays
            List of arrays where each sub-matrix corresponds to entropy exchange.

        Raises
        ------
        AssertionError :
            This does not work for choi-matrices.
        """
        assert self._type == "kraus", "Kraus operators should be provided. Not Choi Matrices."
        # Check if krauss operator matches the krauss operators for channel tensored n times.
        if n != self._current_n:
            self.krauss.update_kraus_operators(n)
        return self.krauss.entropy_exchange(rho, n, adjoint)

    def entropy(self, mat, cut_off=1e-10):
        r"""
        Calculate von neumann entropy by finding all eigenvalues.

        Parameters
        ----------
        mat : array/sparse matrix or list of array
            Either give it a single array/sparse matrix to compute entropy or a list of
            arrays where each one is calculated.

        cut_off : float
            The cut off for zero eigenvalues. Default is 1e-10.

        Returns
        -------
        float :
            The von neumann entropy.
        """
        if self.sparse:
            eigs = np.linalg.eigvalsh(mat)
        else:
            eigs = np.ravel(np.linalg.eigvalsh(mat))
        no_zeros = eigs[np.abs(eigs) > cut_off]
        return -np.sum(no_zeros * np.log2(no_zeros))

    def coherent_information(self, rho, n, regularized=False):
        r"""
        Coherent information of a channel with respect to a density matrix.

        This is defined on a channel N with complementary channel N^c as
        .. math::
            I_c(\rho, N) = S(N(\rho) - S(N^c(\rho))

        Parameters
        ----------
        rho : np.array
            Trace one, positive-semidefinte, hermitian matrix.
        n : int
            The number of times the channel is tensored with itself.
        regularized : bool
            Return the coherent information of rho divided by n.

        Returns
        -------
        float :
            Return the coherent information of a channel with respect to :math:'\rho.'. If
            regularized is true, then coherent information answer is divided by n.

        Notes
        -----
        - If choi matrix is provided, then instead of complementary channel being computing,
        the last term is computated as, .. math::
            I_c(\rho, N) = S(N(\rho) - S((I \otimes N) \Phi)
        where .math.'\Phi' is the purification of .math.'\rho'.
        """
        quantum_chann = self.channel(rho, n)
        if self._type == "kraus":
            entropy_exchange = self.entropy_exchange(rho, n)
        else:
            assert n == 1, "For Choi matrices, n must be equal to one."
            entropy_exchange = self.choi.complementary_channel(rho)

        coherent_info = self.entropy(quantum_chann) - self.entropy(entropy_exchange)
        if regularized:
            return coherent_info / float(n)
        return coherent_info

    def optimize_coherent(self, n, rank, optimizer="diffev", param="overparam", lipschitz=0,
                          use_pool=False, maxiter=50, samples=()):
        r"""
        maximum of coherent information of a channel of all density matrix of fixed rank.

        This is defined on a channel N with complementary channel N^c as
        .. math::
            I_c(N) = \max_{\rho} S(N(\rho) - S(N^c(\rho))

        Parameters
        ----------
        n : int
            The number of times the channel is tensored with itself.
        rank : int
            Rank of the density matrix being optimized.
        optimizer : str
            If "diffev", then optimized using differential evolution.
            If "slsqp", then optimized using slsqp.
        param : str or ParameterizationABC
            If string and "overparam", then optimized using OverParameterization.
            If string and "cholesky", then optimized using CholeskyParameterization.
            If it is a subclass of ParameterizationABC, then optimize using user-specify
                parameterization.
            See "param.py" for more infomation.
        lipschitz : int
            The number of lipschitz sampler to be used for initial guess to be used. If samples
            is empty and optimizer is slsqp, then lipschitz must be greater than zero.
        use_pool : int
            The number of pool processes to use to improve computational speed-up. Should be less
            than the number of CPU cores.
        maxiter : int
            Maximum number of iteration used in the optimizer. Default is 50.
        samples : list
            List of vectors that satisfy the parameterization from "param", that are served as
            initial guesses.

        Returns
        -------
        float :
            Return the coherent information of a channel.

        Notes
        -----
        - Highly recommend optimizing using kraus operators rather than choi-matrices.
        - With choi-matrices, the channel can't be tensored, so n must be equal to one.
        - Highly recommend using lipschitz to find initial guesses rather than using
            LatinHypercube in differetial evolution or using one's own sampler from 'samples'.
        - If user has their own parameterization scheme, then it must be a sub-class of
        ParameterizationABC from 'param.py' file and provided in the 'param' attribute.
        - For large n, it might be more suitable to use sparse kraus operators, however this
        greatly increases computational time.
        - The rank should always be less than or equal to '(dim_in ** numb_qubits[0])**n',
        it is highly recommended to make the rank maximal, as parameterization is unique and highly
        suspected that maximal rank of density state is the global maxima of coherent
        information. Another reason to use maximal rank, is that positive-definite matrices are
        dense in space of positive-semidefinite matrices, hence the optimal answer found will have
        eigenvalues close to zero and will model any rank density matrix.
        - A good strategy is to use lipschitz sampler with SLSQP. It has comparable accuracy to
        differential_evolution with LatinHypercube sampler. Increase use_pool, increases
        computation-speed-up at the cost of using cpu-cores.
        """
        if self._type == "choi":
            assert n == 1, "For Choi-matrices only n=1 can be evaluated. Turn to Kraus Operators."
        else:
            # Update the "current_n" to correspond to user-provided n.
            if n != self._current_n:
                self.krauss.update_kraus_operators(n)
        return optimize_procedure(self, n=n, rank=rank, optimizer=optimizer, param=param,
                                  objective="coherent", lipschitz=lipschitz, use_pool=use_pool,
                                  maxiter=maxiter, samples=samples)

    def optimize_fidelity(self, n, optimizer="diffev", param="overparam", lipschitz=0,
                          use_pool=False, maxiter=50, samples=()):
        r"""
        Optimizes the minimum fidelity of a channel over all pure states.

        Parameters
        ----------
        n : int
            The number of times the channel is tensored with itself.
        rank : int
            Rank of the density matrix being optimized.
        optimizer : str
            If "diffev", then optimized using differential evolution.
            If "slsqp", then optimized using slsqp.
        param : str or ParameterizationABC
            If string and "overparam", then optimized using OverParameterization.
            If string and "cholesky", then optimized using CholeskyParameterization.
            If it is a subclass of ParameterizationABC, then optimize using user-specify
                parameterization.
            See "param.py" for more infomation.
        lipschitz : int
            The number of lipschitz sampler to be used for initial guess to be used. If samples
            is empty and optimizer is slsqp, then lipschitz must be greater than zero.
        use_pool : int
            The number of pool processes to use to improve computational speed-up. Should be less
            than the number of CPU cores.
        maxiter : int
            Maximum number of iteration used in the optimizer. Default is 50.
        samples : list
            List of vectors that satisfy the parameterization from "param", that are served as
            initial guesses.

        Returns
        -------
        float :
            Return the mininum fidelity of a channel.

        Notes
        -----
        - Optimization of minimum fidelity is over pure states and hence rank one density matrices.
        """
        if self._type == "choi":
            assert n == 1, "For Choi-matrices only n=1 can be evaluated. Turn to Kraus Operators."
        # Update the "current_n" to correspond to user-provided n.
        else:
            if n != self._current_n:
                self.krauss.update_kraus_operators(n)
        return optimize_procedure(self, n=n, rank=1, optimizer=optimizer, param=param,
                                  objective="fidelity", lipschitz=lipschitz, use_pool=use_pool,
                                  maxiter=maxiter, samples=samples)

    @staticmethod
    def concatenate_two_channels(channel1, channel2):
        r"""
        Serial Concantenation of two channels, A, B, to produce a third channel A \circ B.

        Parameters
        ----------
        channel1 : AnalyticQChan
            Channel being concatenated on left-side.

        channel2 : AnalyticQChan
            Channel being concantenated on right-side.

        Returns
        -------
        AnalyticQChan :
            Channel A \circ B.
        """
        pass

    @staticmethod
    def one_qubit_channel(l_vec, t_vec, pauli_errors=False):
        r"""
        Choi matrix based on the one qubit parameterization.

        Parameters
        ----------
        l_vec : np.ndarray(M,)
            [lx, ly, lz], the lambdas parameters. If pauli_errors is true, then l_vec denotes the
            set of pauli errors p_x, p_y, p_z, where p_x denotes the probability of bit-flip X
            error, and p_y, p_z denotes the probability of Y and phase-flip error Z, respectively.

        t_vec : np.ndarray(M,)
            [Tx, Ty, TZ] the non-unital parameters, each correponding to a translation of the
            maximally mixed state by the quantum channel.

        pauli_errors : bool
            True if l_vec is the set of pauli errors ie l_vec = [px, py, pz], where px is the
            probability of pauli-X error, etc.

        Returns
        -------
        AnalyticQChan :
            Returns AnalyticQChan based on the choi matrix of the one-qubit channel.

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
        choi_obj = ChoiQutip.one_qubit_choi_matrix(l_vec, t_vec, pauli_errors)
        return AnalyticQChan(choi_obj.choi, [1, 1], 2, 2)


from projectq import MainEngine  # import the main compiler engine
from projectq.ops import H, Measure  # import the operations we want to perform (Hadamard and measurement)


class QDeviceChannel():
    r""""""
    def __init__(self, numb_qubits, circuit, engine=MainEngine, real_device=False):
        r"""

        Parameters
        ----------
        numb_qubits : int
            Number of qubits

        circuit : callable
            Should take as input the engine from 'ProjectQ.cengines'.

        engine :
            Should be one of the following types from 'ProjectQ.cengines'. Should
            be of type "BasicEngine" from ProjectQ. See their documentation
            for more details,
            "https://projectq.readthedocs.io/en/latest/projectq.cengines.html".

        real_device : bool
            Boolean indicating whether it's being runned on a real quantum device.

        Returns
        -------

        """
        self.engine = engine
        self.circuit = circuit
        super(QChanABC).__init__(numb_qubits)

    def channel():
        pass
