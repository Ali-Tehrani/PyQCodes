r"""
The MIT License

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
from abc import abstractstaticmethod

r"""
Contains static classes for parameterization of Density States intended for optimization.

All routines are allowed to control the rank of the density matrix. This is due to the fact that
the density matrices are not a smooth manifold but it's decomposed into fixed rank are smooth 
manifolds.

Three Routines That Are of Interest. 

ParameterizationABC : 
    Abstract base class for parameterization of density matrices.
    This is intended if one wants their own parameterization of density matrices.
OverParam : 
    Child class of ParameterizationABC. Models a state as the Cholesky decomposition L^\dagger L,
    where L is any (rank times m) size matrix. Use this for small systems.
CholeskyParam :
    Child class of ParameterizationABC. Models a state as the Cholesky decomposition L^\dagger L,
    where L is a lower triangular, (rank times m) size matrix . Use this for larger systems.
"""

__all__ = ["StandardBasis", "OverParam", "CholeskyParam", "ParameterizationABC"]


class ParameterizationABC:
    @abstractstaticmethod
    def numb_variables(nsize, rank):
        raise NotImplementedError

    @abstractstaticmethod
    def bounds(nsize):
        raise NotImplementedError

    @abstractstaticmethod
    def rho_from_vec(vec, nsize):
        raise NotImplementedError

    @abstractstaticmethod
    def vec_from_rho(rho, nsize):
        pass

    @abstractstaticmethod
    def random_vectors(nsize, rank):
        # Used for the lipschitz sampler.
        pass


class StandardBasis(ParameterizationABC):
    r"""
    Static class for parameterization of hermitian, complex-matrices.

    For a given d times d complex-matrix, total of d^2 number of real-variables.
    Each variable is further restricted in [-1, 1]. Note that no checks
    are being made to verify it satisfies these conditions and must be determined by user.
    Not recommended for optimization purpose, it's advantage is it's simplicity.

    Methods
    -------
    numb_variables :
        Return number of variables needed to represent d times d hermitian complex-matrix.
    bounds :
        Returns the range of the variables.
    rho_from_vec :
        From vector return the d times d hermitian complex-matrix.
    vec_from_rho :
        From hermitian complex-matrix, return d^2 real-variable vector.

    Examples
    --------
    # Vector to Rho
    vec = np.array([0.5, 0.2, 0.1, 0.5])
    dim = 2
    rho = StandardBasis.rho_from_vec(rho, 2)
    # rho = np.array([[0.5, 0.2 + 0.1j], [0.2 - 0.1j, 0.5]])

    # Rho to Vector
    rho = np.array([[0.25, 0.2 + 0.1j, 0.3],
                    [0.5 - 0.1j, 0.25, 0.23 + 0.1j],
                    [0.3, 0.23 - 0.1j, 0.5]])
    dim = 3
    vec = StandardBasis.vec_from_rho(rho, dim)
    # vec = [0.25, 0.2, 0.1, 0.3, 0, 0.25, 0.23, 0.1, 0.5]

    """
    @staticmethod
    def numb_variables(d):
        # Number of variables of a dxd hermitian complex-matrix.
        return d**2

    @staticmethod
    def bounds(d):
        # Bounds on the range of the hermitian complex-matrix of size d x d.
        return [(-1, 1)] * StandardBasis.numb_variables(d)

    @staticmethod
    def rho_from_vec(vec, nrows):
        r"""
        Convert nrows^2 real-variables to a hermitian matrix.

        Conversion is done row by row basis.

        Parameters
        ----------
        vec : array
            nrows^2 of real-variables.
        nrows: int
            Number of rows and columns of the hermitian matrix.

        Returns
        -------
        array :
            Hermitian complex-matrix with number of columns/rows is nrows.
        """
        assert len(vec) == nrows * nrows
        rho = np.zeros((nrows, nrows), dtype=np.complex128)
        counter = 0
        for i in range(0, nrows):
            for j in range(i, nrows):
                rho[i, j] += complex(vec[counter], 0)
                # Diagonal matrix-elements are always real, so move to next variable..
                if i == j:
                    counter += 1
                # Off-diagonal matrix-elements.
                if i != j:
                    # Complex numbers are represented by two real-variables.
                    rho[i, j] += complex(0, vec[counter + 1])
                    # Update the other symmetric off-diagonal matrix element.
                    rho[j, i] += complex(vec[counter], 0)
                    rho[j, i] += complex(0, -vec[counter + 1])
                    # Move up two steps/variables.
                    counter += 2

        return rho

    @staticmethod
    def vec_from_rho(rho, nrows):
        r"""
        Convert a nrows^2 vector of real-variables to a hermitian complex-matrix.

        Parameters
        ----------
        rho : array
            A hermitian matrix of size nrows * nrows.

        nrows  : int
            The number of columns of rho.

        Returns
        -------
        array :
            One dimensional array of real-variables of size nrows^2.
        """
        assert rho.shape == (nrows, nrows)
        vec = np.zeros(nrows ** 2)
        counter = 0
        for i in range(0, nrows):
            for j in range(i, nrows):
                vec[counter] = np.real(rho[i, j])
                counter += 1
                if i != j:
                    vec[counter] = np.imag(rho[i, j])
                    counter += 1
        return vec


class OverParam(ParameterizationABC):
    r"""
    Over-parameterization of hermitian, positive-semi definite, trace-one complex-matrix.

    Every hermitian, positive-semi definite matrix M can be written non-uniquely as A *
    A^{\dagger}. The over-parameterization is a standard-basis parameterization of matrix A,
    rather than M.

    Methods
    -------
    random_vectors(dist="normal") :
        Return a random vector based on probability distribution, default is normal distribution and
        the other option is "uniform" distribution between -1 to 1.
    numb_variables :
        Returns the number of variables to represent a hermitian, trace-one complex-matrix.
    bounds :
        Return the range of variable, defualt is from -1 to 1.
    rho_from_vec(rank=None):
        Return hermitian, trace-one complex-matrix from real-variable vectors. If rank is none,
        then parameterization assumes full-rank matrix.
    vec_from_rho :
        Return real-variable vectors from hermitian, trace-one complex-matrix.

    Examples
    --------
    # Obtain Vector from rho
    nsize = 2  # Number of columns/rows.
    rho = np.array([[0.5, 0], [0, 0.5]])
    A = np.array([[np.sqrt(0.5), 0], [0, np.sqrt(0.5)]])
    vec = OverParam.vec_from_rho(rho, nsize=2, rank=2)
    # vec should be [np.sqrt(0.5), 0, 0, 0, 0, 0, np.sqrt(0.5), 0]

    """
    def random_vectors(nsize, rank, dist="normal"):
        r"""
        Random vectors from specified distribution.

        Parameters
        ----------
        nsize : int
            Number of rows (or columns) of the random density state.
        rank : int
            Rank of the random density state.
        dist : str
            Produces random density state either from normal distribution "normal"
            with mean zero and variance 1 or uniform distribution "uniform" from -1 to 1.

        Returns
        -------
        list :
            Gets random vector from the specified distribution.

        Notes
        -----
        This is used in the Lipschitz Sampler procedure for finding initial guesses.
        """
        if dist == "normal":
            random_vecs = np.random.normal(0., 1.,
                                           size=(OverParam.numb_variables(nsize, rank)))
        elif dist == "uniform":
            random_vecs = np.random.uniform(-1, 1.,
                                            size=(OverParam.numb_variables(nsize, rank)))
        else:
            raise TypeError("Wrong distribution parameter")
        return random_vecs

    @staticmethod
    def numb_variables(nsize, rank):
        # The factor two is for representing complex-numbers.
        # The matrix A is size (nsize, rank) and hence has nrow * rank number of matrix entries.
        return 2 * rank * nsize

    @staticmethod
    def bounds(nsize, rank):
        r"""
        Return the lower and upper bounds of each variable for use of optimization algorithms.

        Matrix entries of a hermitian, trace-one matrix is bounded above by one because the
        eigenvalues are bounded above by one. Hence the lower and upper bound are justified to be
        one.

        Parameters
        ----------
        nsize : int
            Number of rows (or columns) of the hermitian, trace-one matrix.
        rank : int
            The Rank of the desired density state.

        Returns
        -------
        list of tuples:
            Each entry is (-1., 1.), where are the bounds for each variable.
        """
        # The bound are like this for l-bfgs solver that go to the boundary
        return [(-1, 1)] * OverParam.numb_variables(nsize, rank)

    @staticmethod
    def rho_from_vec(vec, nsize, rank=None):
        r"""
        Creates a density state from the over-parameterization method.

        Every hermitian matrix M of size (nsize, nsize) can be written as A * A^{\dagger} .
        The vec parameterization the matrix A which has matrix size of (rank, nsize).

        Parameters
        ----------
        vec : array
            One-dimensional array of real-variables vector.
        nsize : int
            Number of row (or column) of the hermitian, trace-one hermitian matrix.
        rank : int
            The rank of hermitian, trace-one hermitian matrix.

        Returns
        -------
        array :
            Returns a hermitian, trace-one matrix of rank "rank".

        """
        # Note that matrix entries are bounded above by one, because the eigenvalues are bounded
        # above by one.
        # assert np.all((-1. <= vec) & (vec <= 1.))
        # assert np.all(1. - np.abs(vec) > -1e-7)
        if rank is not None:
            assert rank <= nsize
            assert len(vec) == 2 * rank * nsize
        elif rank is None:
            rank = len(vec) // (2 * nsize)

        # Get standard-basis representation of matrix A.
        complex_vec = [complex(vec[x], vec[x + 1]) for x in range(0, len(vec), 2)]
        a = np.reshape(complex_vec, (nsize, rank))
        rho = np.dot(a, np.conj(a).T)
        # Divide by trace, to make it trace one.
        return rho / np.trace(rho)

    @staticmethod
    def vec_from_rho(rho, nsize, rank):
        r"""
        Return hermitian, trace-one hermitian matrix from a real-vector of size 2 * nsize * rank.

        This is not unique and multiple answers is possible. This method is done via
        diagonalization. The matrix A is written as U * \sqrt{D}, where U is the eigenvectors and
        D is the diagonal matrix of eigenvalues.

        Parameters
        ----------
        rho : array
            Hermitian, trace-one complex-matrix.
        nsize : int
            Number of rows (or columns) of rho.
        rank : int
            The rank of rho.

        Returns
        -------
        array :
            One dimensional array of real-variables if size 2 * nsize * rank.
        """
        # Get eigenvalues D and eigenvectors U and truncate it if it's close to zero.
        eigs, evecs = np.linalg.eigh(rho)
        eigs[np.abs(eigs) < 1e-5] = 0
        # Create matrix A by = U * sqrt(D)
        # Eigenvalues are increasing so we can ignore up-to number of nonzero eigenvalues.
        numb_nonzero_eigs = nsize - rank
        a = np.ravel(evecs[:, numb_nonzero_eigs:].dot(np.diag(np.sqrt(eigs[numb_nonzero_eigs:]))))
        vec = np.zeros(2 * nsize * rank)
        # Update vector
        counter = 0  # Counter for updating vector
        for i in range(0, len(a)):  # Counter for matrix entry of a.
            vec[counter] = np.real(a[i])
            vec[counter + 1] = np.imag(a[i])
            counter += 2
        return vec


class CholeskyParam(ParameterizationABC):
    r"""
    Cholesky parameterization of hermitian, positive-semidefinite (PSD) trace-one matrix.

    The Cholesky Decomposition states that any hermitian, positive-semidefinite, matrix can be
    written as L^{\dagger}* L, where L is a lower-triangular matrix. The matrix L is called a
    cholesky factor. The cholesky parameterization is a standard-basis parameterization of L.
    The downside of this parameterization is that it is non-unique.

    Methods
    -------
    random_vectors(dist="normal") :
        Return a random vector based on probability distribution, default is normal distribution and
        the other option is "uniform" distribution between -1 to 1.
    numb_variables :
        Returns the number of variables to represent a hermitian, positive-semidefinite (PSD),
        trace-one complex-matrix.
    bounds :
        Return the range of variable, defualt is #TODO: Fill this out.
    rho_from_vec(rank=None):
        Return hermitian, PSD, trace-one complex-matrix from real-variable vectors. If rank is none,
        then parameterization assumes full-rank matrix.
    vec_from_rho :
        Return real-variable vectors from hermitian, PSD, trace-one complex-matrix.

    References
    ----------
    .. [1] Pinheiro, J. C., & Bates, D. M. (1996).  Unconstrained parametrizations for
           variance-covariance matrices. Statistics and computing, 6(3), 289-296.

    Examples
    --------
    # Obtain Vector from rho
    nsize = 2  # Number of columns/rows.
    rho = np.array([[0.5, 0], [0, 0.5]])
    A = np.array([[np.sqrt(0.5), 0], [0, np.sqrt(0.5)]])
    vec = CholeskyParam.vec_from_rho(rho, nsize=2, rank=2)
    # Note that it is non-unique parameterization.

    """
    @staticmethod
    def random_vectors(nsize, rank, dist="uniform", l_bnd=-1, u_bnd=1.):
        r"""
        Return a cholesky-parameterized vector based on a random density state.

        Parameters
        ----------
        nsize : int
            Dimension of the random density state.
        rank : int
            Rank of the random density state.
        dist : str
            Produces random density state either from normal distribution "normal"
             with mean zero and variance 1 or uniform distribution "uniform" from l_bnd to u_bnd.
        l_bnd: float
            The lower bound of each non-diagonal, real variable. Default is -1.
        u_bnd : float
            The upper bound of each real variable. Default is 1. Should always be greater than zero.

        Returns
        -------
        list
            Gets random vector from the specified distribution.

        Notes
        -----
        The normal distribution is scaled so that it is between u_bnd. Highly Recommend Uniform
        distribution. This is used in the Lipschitz Sampler procedure for finding initial guesses.
        """
        assert u_bnd >= 0.
        numb_vars = CholeskyParam.numb_variables(nsize, rank)
        if dist == "normal":
            rank_eigs = np.abs(np.random.normal(u_bnd / 2., u_bnd / 4., size=(rank)))
            rank_eigs /= np.max(rank_eigs)
            random_vecs = np.random.normal((u_bnd - l_bnd) / 2., (u_bnd - l_bnd) / 4.,
                                           size=(numb_vars - rank))
        elif dist == "uniform":
            rank_eigs = np.random.uniform(0., u_bnd, size=(rank))
            random_vecs = np.random.uniform(l_bnd, u_bnd, size=(numb_vars - rank))
        else:
            raise TypeError("Wrong distribution parameter")
        random_vecs = np.append(rank_eigs, random_vecs)
        return random_vecs

    @staticmethod
    def numb_variables(nsize, rank):
        # The number of varibles needed to represent the cholesky factor L.
        return rank * (rank + 1) + 2 * (nsize - rank) * rank - rank

    @staticmethod
    def bounds(nsize, rank, l_bnd=-1., u_bnd=1.):
        r"""
        Get the upper and lower bounds for each variable in the Cholesky Parameterization.

        Parameters
        ----------
        nsize : int
            The number of rows (or columns) of the trace-one, hermitian, PSD matrix. For qubit
            density states, nsize is 2^n for some positive, integer n.
        rank : int
            The Rank of the desired density state.
        l_bnd: float
            The lower bound of each non-diagonal, real variable. Default is -1.
        u_bnd : float
            The upper bound of each real variable. Default is 1. Should always be greater than zero.

        Returns
        -------
        list of tuples :
            Each entry is a tuple consisting of (l_bnd, u_bnd) for each real variable.
            If entry is a diagonal entry then it has form of (0, u_bnd).

        Notes
        -----
        In the Cholesky Decomposition, the lower triangular matrix is required
        to have positive diagonal entries. The lower and upper bound of (-1, 1)
        follows from a density matrix having trace one and positive-eigenvalues.
        """
        numb_real_vars = CholeskyParam.numb_variables(nsize, rank)
        # First rank elements are the diagonal elements.
        # The rest of the elements are the off-diagonal components
        bounds = [(0., u_bnd)] * rank + [(l_bnd, u_bnd)] * (numb_real_vars - rank)
        return bounds

    @staticmethod
    def rho_from_vec(vec, nsize, rank=None):
        r"""
        Hermitian, trace-one, positive-semidefinite (PSD) complex-matrix from vector with rank.

        Parameters
        ----------
        rho : array
            Hermitian, trace-one, PSD, complex-matrix.
        nsize : int
            Number of rows (or columns) of rho.
        rank : int
            The rank of rho. if rank is "None', then it is assumed to be full-rank ie equal to
            nsize.

        Returns
        -------
        array :
            Returns complex matrix of shape (nsize, nsize) that is hermitian, trace-one and
            positive-semidefinite with specified rank.

        Notes
        -----
        Every hermitian matrix can be written as L * L^{\dagger}, where L is a lower triangular
        matrix with real and positive diagonal entries. The vec is a parameterization of the
        lower-triangular complex-matrix L from the left to right. Note that for a PSD matrix,
        then diagonal elements of L is positive. The first n-elements of vec are the diagonal
        elements of L, the next set of elements is the real component and imaginery component of
        the off-diagonal components.

        References
        ----------
        .. [1] Pinheiro, J. C., & Bates, D. M. (1996).  Unconstrained parametrizations for
               variance-covariance matrices. Statistics and computing, 6(3), 289-296.

        """
        if rank is None:
            rank = nsize

        assert 1 <= rank <= nsize
        assert len(vec) == CholeskyParam.numb_variables(nsize, rank)
        # Compute Cholesky Factor L.
        choles_fac = np.zeros((nsize, rank), dtype=np.complex128)
        # Get non-diagonal elements as complex numbers.
        complex_vec = [complex(vec[x], vec[x + 1]) for x in range(rank, len(vec), 2)]
        # Update non-diagonal components
        choles_fac[np.tril_indices(nsize, k=-1, m=rank)] = complex_vec
        # Update diagonal components (they are always real and positive).
        choles_fac[np.diag_indices(rank)] = vec[:rank]
        product = np.dot(choles_fac, np.conj(choles_fac).T)
        return product / np.trace(product)

    @staticmethod
    def vec_from_rho(rho, nsize, rank=None):
        r"""
        Convert real-vectors to a hermitian, trace-one, PSD matrix via Cholesky Parameterization.

        Parameters
        ----------
        rho : array
            Hermitian, trace-one, PSD, complex-matrix.
        nsize : int
            Number of rows (or columns) of rho.
        rank : int
            The rank of rho. if rank is "None', then it is assumed to be full-rank ie equal to
            nsize.

        Returns
        -------
        array :
            One-dimensional real-vector that parameterizes the matrix "rho".

        Notes
        -----
        Every hermitian matrix can be written as L * L^{\dagger}, where L is a lower triangular
        matrix with real and positive diagonal entries. The vec is a parameterization of the
        lower-triangular complex-matrix L from the left to right. Note that for a PSD matrix,
        then diagonal elements of L is positive. The first n-elements of vec are the diagonal
        elements of L, the next set of elements is the real component and imaginery component of
        the off-diagonal components.

        References
        ----------
        .. [1] Pinheiro, J. C., & Bates, D. M. (1996).  Unconstrained parametrizations for
                variance-covariance matrices. Statistics and computing, 6(3), 289-296.
        """
        if rank is None:
            rank = nsize
        assert 1 <= rank <= nsize
        cholesky_fac = np.linalg.cholesky(rho)
        # First set of elements is the diagonal elements (always real).
        vec = np.diag(np.real(cholesky_fac))
        # Add real Off-diagonal elements
        vec = np.append(vec, np.real(cholesky_fac[np.tril_indices(nsize, k=-1, m=rank)]))
        # Add imaginary Off-diagonal elements
        vec = np.append(vec, np.imag(cholesky_fac[np.tril_indices(nsize, k=-1, m=rank)]))
        return vec
