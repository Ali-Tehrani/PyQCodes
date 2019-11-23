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
from PyQCodes.chan._kraus import SparseKraus, DenseKraus
from PyQCodes.chan._choi import ChoiQutip
from PyQCodes.chan._coherent import optimize_procedure

import numpy as np
from scipy.linalg import sqrtm
from sparse import SparseArray

r"""
Contains AnalyticQChan: Representing Quantum Channels Modeled as Kraus Operators or Choi Matrices.
"""

__all__ = ["AnalyticQChan"]


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
        'operator' attribute.
    kraus : list
        Returns the kraus operators when n=1.

    Methods
    -------
    fidelity_two_states :
        The fidelity between two density states.
    average_entanglement_fidelity :
        The average entanglement fidelity with respect to a ensemble of states.
    channel :
        Calculates the quantum channel on a density matrix.
    entropy_exchange :
        Calculates the complementary channel/entropy exchange matrix on a density matrix.
    entropy :
        Calculates the entropy (log base 2) with respect to a density matrix.
    coherent_information :
        Calculates the coherent information with respect to a density matrix.
    optimize_coherent :
        Maximizes the coherent information of a given channel.
    optimize_fidelity :
        Minimizes the fidelity over the space of pure-states of a given channel.

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

    >> numb_qubits = [1, 1]  # Accepts one qubits and outputs one qubits.
    >> dim_in = 2  # Dimension of single qubit hilbert space is 2.
    >> dim_out = 2  # Dimension of single qubit hilbert space after the bit-flip channel is 2.

    >> p = 0.25  # Probability of error
    >> identity = np.sqrt(p) * np.eye(2)
    >> bit_flip = np.sqrt(1 - p) * np.array([[0., 1.], [1., 0.]])
    >> kraus_ops = [identity, bit_flip]
    >> channel = AnalyticQChan(kraus_ops, numb_qubits, dim_in, dim_out)

    Optimize the coherent information using "differential evolution" and parameterization
    of density matrix with OverParameterization.
    >> res = channel.optimize_coherent(n=2, rank=4, optimizer="diffev", param="overparam",
    >>                                maxiter=200)
    >> print(res)

    """

    def __init__(self, operator, numb_qubits, dim_in, dim_out, orthogonal_krauss=(), sparse=False):
        r"""
        Construct the AnalyticQChan class.

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
        Constructing a simple qubit channel
        >> numb_qubits = [1, 1]  # Accepts one qubits and outputs one qubits.
        >> dim_in = 2  # Dimension of single qubit hilbert space is 2.
        >> dim_out = 2  # Dimension of single qubit hilbert space after the qubit channel is 2.

        >> p = 0.25  # Probability of error
        >> identity = np.sqrt(p) * np.eye(2)
        >> bit_flip = np.sqrt(1 - p) * np.array([[0., 1.], [1., 0.]])
        >> kraus_ops = [identity, bit_flip]
        >> channel = AnalyticQChan(kraus_ops, numb_qubits, dim_in, dim_out, sparse=True)

        Constructing the erasure channel where dimension of output hilbert space is larger.
        >> numb_qubits = [1, 1]
        >> dim_in = 2
        >> dim_out = 3  # Erasure channel adds a extra dimension to output of the channel.

        >> p = 0.25  # Probability of no error
        >> identity = np.sqrt(p) * np.array([[1., 0.], [0., 1.], [0., 0.]])
        >> e1 = np.sqrt(1 - p) * np.array([[0., 0.], [0., 0.], [1., 0.]])
        >> e2 = np.sqrt(1 - p) * np.array([[0., 0.], [0., 0.], [0., 1.]])

        >> kraus_ops = [identity, e1, e2]
        The indices {0} of kraus ops is orthogonal to kraus ops indices to {1, 2}
        >> orthogonal = [1]
        >> channel = AnalyticQChan(kraus_ops, numb_qubits, dim_in, dim_out,
        >>                         orthogonal_krauss=orthogonal)

        """
        if not (isinstance(operator, (list, np.ndarray, SparseArray))):
            # Test if it is kraus operator or choi type.
            raise TypeError("Operator argument should be a list, ndarray or SparseArray.")

        # If Operator is Kraus Type
        self.sparse = sparse
        if (isinstance(operator, (np.ndarray, SparseArray)) and operator.ndim == 3) or \
                isinstance(operator, list):
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

    @property
    def nth_kraus_operators(self):
        r"""
        Get the Kraus operators for the nth-channel based on "current_n" attribute.

        Returns
        -------
        np.ndarray :
            Returns three-dimensional (k, i, j) numpy array, where each kth array is a kraus
            operator representing the channel tensored n times.

        Raises
        ------
        AssertionError :
            The channel must be kraus operator style, not choi matrix.

        """
        # Get kraus operators for the nth-channel use.
        assert self._type == "kraus", "Function only works with kraus operators."
        return self.krauss.nth_kraus_ops

    @property
    def input_dimension(self):
        r"""
        Return the dimension of a single particle of the input to the power of number of particles.

        Should be number of rows/columns of density matrix, that is about to go through channel.
        """
        if self._type == "kraus":
            return self.krauss.input_dim**self._current_n
        else:
            return self.choi.input_dim

    @property
    def output_dimension(self):
        r"""
        Return the dimension of a single particle of the ouput to the power of number of particles.

        Should be number of rows/columns of density matrix, after going through channel.
        """
        if self._type == "kraus":
            return self.krauss.output_dim**self._current_n
        else:
            return self.choi.output_dim

    @property
    def _current_n(self):
        if self._type == 'kraus':
            return self.krauss.current_n
        else:
            # Choi matrices can only have n=1.
            return 1

    @property
    def numb_krauss(self):
        r"""Return the number of kraus operators for the single (n=1) channel."""
        assert self._type == "kraus", "Function only works with kraus operators."
        return self.krauss.numb_krauss

    @property
    def kraus(self):
        r"""Return the kraus operators for the single (n=1) initial channel."""
        if self._type == "kraus":
            if self.sparse:
                return self.krauss.kraus_ops.todense()
            else:
                return self.krauss.kraus_ops
        else:
            return self.choi.kraus_operators()

    @property
    def dim_in(self):
        r"""Return the dimension of a single particle that is the input of the channel."""
        if self._type == "kraus":
            return self.krauss.dim_in
        else:
            return self.choi.dim_in

    @property
    def dim_out(self):
        r"""Return the dimension of a single particle that is the output of the channel."""
        if self._type == "kraus":
            return self.krauss.dim_out
        else:
            return self.choi.dim_out

    @property
    def numb_qubits(self):
        r"""Return the number of particles."""
        if self._type == "kraus":
            return self.krauss.numb_qubits
        else:
            return self.choi.numb_qubits

    def __add__(self, other):
        r"""
        Return channel A + B, where B is applied first then A is applied next.

        Parameters
        ----------
        other : AnalyticQChan
            Quantum channel object.

        Returns
        -------
        AnalyticQChan :
            Returns the channel A + B, where B is applied first then A is applied next.

        Raises
        ------
        TypeError :
            Returns error if the dimension of A does not match the dimension B.

        """
        assert self._type == "kraus", "Only works for kraus operators."
        issparse = self.sparse or other.sparse
        if issparse:
            new_kraus_ops = SparseKraus.serial_concatenate(self.kraus, other.kraus)
        else:
            new_kraus_ops = DenseKraus.serial_concatenate(self.kraus, other.kraus)
        numb_qubits = [other.numb_qubits[0], self.numb_qubits[1]]
        dim_in = other.dim_in
        dim_out = self.dim_out
        return AnalyticQChan(new_kraus_ops, numb_qubits, dim_in, dim_out, sparse=issparse)

    def __mul__(self, other):
        r"""
        Parallel Concatenate two channels A \otimes B.

        Parameters
        ----------
        other : AnalyticQChan
            The channel object B.

        Returns
        -------
        AnalyticQChan :
            Returns a new channel that models A tensored with B.

        Notes
        -----
        - The number of particles set to one particle and one particle for output.

        """
        assert self._type == "kraus", "Only works for kraus operators."
        issparse = self.sparse or other.sparse
        if issparse:
            new_kraus_ops = SparseKraus.parallel_concatenate(self.kraus, other.kraus)
        else:
            new_kraus_ops = DenseKraus.parallel_concatenate(self.kraus, other.kraus)
        numb_qubits = [1, 1]
        dim_in = self.dim_in * other.dim_in
        dim_out = self.dim_out * other.dim_out
        return AnalyticQChan(new_kraus_ops, numb_qubits, dim_in, dim_out, sparse=issparse)

    def _is_qubit_channel(self):
        r"""Return true if it is qubit channel."""
        if self.input_dimension == 2 and self.output_dimension == 2:
            return True
        return False

    def _update_channel_tensored(self, n):
        r"""Update the channel so that it becomes the channel tensored n-times."""
        if self._type == "choi":
            assert n == 1, "For Choi-matrices only n=1 can be evaluated. Turn to Kraus Operators."
        # Update the "current_n" to correspond to user-provided n.
        else:
            if n != self._current_n:
                self.krauss.update_kraus_operators(n)

    def fidelity_two_states(self, rho, n):
        r"""
        Calculate the fidelity between rho and the channel evaluated on rho.

        The fidelity is defined as :
        .. math:: F(\rho, \mathcal{N}(\rho)) = Tr(\sqrt{\sqrt{\rho}\mathcal{N}(\rho)\sqrt{\rho}}).

        Parameters
        ----------
        rho : np.ndarray
            The density matrix.
        n : int
            The number of times the channel is tensored.

        Returns
        -------
        float :
            The fidelity between rho and the channel evaluated on rho.

        """
        chan = self.channel(rho, n)
        sqrt_rho = sqrtm(rho)
        fidel = np.trace(sqrtm(sqrt_rho.dot(chan).dot(sqrt_rho)))
        assert np.abs(np.imag(fidel)) < 1e-5
        return np.real(fidel)

    def average_entanglement_fidelity(self, probs, states):
        r"""
        Calculate average entanglement fidelity of an given ensemble.

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
        Compute the entropy exchange matrix (complementary channel) given kraus operators.

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
                          use_pool=False, maxiter=50, samples=(), disp=False, regularized=False):
        r"""
        Maximum of coherent information of a channel of all density matrix of fixed rank.

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
        disp : bool
            Print and display during the optimization procedure. Default is false.
        regularized : bool
            Return the coherent information of rho divided by n.

        Returns
        -------
        dict :
        The result is a dictionary with fields:

            optimal_rho : np.ndarray
                The density matrix of the optimal solution.
            optimal_val : float
                The optimal value of either coherent information or fidelity.
            method : str
                Either diffev or slsqp.
            success : bool
                True if optimizer converges.
            objective : str
                Either coherent or fidelity
            lipschitz : bool
                True if uses lipschitz properties to find initial guesses.

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
        self._update_channel_tensored(n)
        result = optimize_procedure(self, n=n, rank=rank, optimizer=optimizer, param=param,
                                    objective="coherent", lipschitz=lipschitz, use_pool=use_pool,
                                    maxiter=maxiter, samples=samples)
        if regularized:
            result["optimal_val"] /= float(n)
        return result

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
        dict :
        The result is a dictionary with fields:

            optimal_rho : np.ndarray
                The density matrix of the optimal solution.
            optimal_val : float
                The optimal value of either coherent information or fidelity.
            method : str
                Either diffev or slsqp.
            success : bool
                True if optimizer converges.
            objective : str
                Either coherent or fidelity
            lipschitz : bool
                True if uses lipschitz properties to find initial guesses.

        Notes
        -----
        - Optimization of minimum fidelity is over pure states and hence rank one density matrices.

        """
        self._update_channel_tensored(n)
        result = optimize_procedure(self, n=n, rank=1, optimizer=optimizer, param=param,
                                    objective="fidelity", lipschitz=lipschitz, use_pool=use_pool,
                                    maxiter=maxiter, samples=samples)
        return result
