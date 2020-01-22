r"""
The MIT License

Copyright (c) 2019-Present PyQCodes - Software for investigating
coherent information and optimization-based quantum error-correcting codes.
PyQCodes is jointly owned equally by the University of Guelph (and its employees)
and Huawei, funded through the Huawei Innovation Research.

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
from PyQCodes.chan.param import StandardBasis, OverParam, CholeskyParam, ParameterizationABC
from qutip import rand_dm_ginibre
import numpy as np

from numpy.testing import assert_array_almost_equal, assert_raises
r"""Tests for "PyQCodes.param.py"."""


def test_standard_basis():
    r"""Tests for "PyQCodes.param.StandardBasis" class."""
    # Test going from vectors to density states.
    numb_tests = 1000
    for n in range(1, 3):
        for _ in range(0, numb_tests):
            v = np.random.uniform(-1., 1., 2**(n*2))
            actual_rho = StandardBasis.rho_from_vec(v, 2 ** n)
            if n == 1:
                desired_rho = np.array([[complex(v[0], 0.), complex(v[1], v[2])],
                                        [complex(v[1], -v[2]), complex(v[3], 0.)]])
            elif n == 2:
                desired_rho = np.array([[complex(v[0], 0.), complex(v[1], v[2]),
                                         complex(v[3], v[4]), complex(v[5], v[6])],
                                        [complex(v[1], -v[2]), complex(v[7], 0.),
                                         complex(v[8], v[9]), complex(v[10], v[11])],
                                        [complex(v[3], -v[4]), complex(v[8], -v[9]),
                                         complex(v[12], 0.), complex(v[13], v[14])],
                                        [complex(v[5], -v[6]), complex(v[10], -v[11]),
                                         complex(v[13], -v[14]), complex(v[15], 0.)]])
            assert_array_almost_equal(actual_rho, desired_rho)

    # Test going from rho to vector to rho.
    for n in range(1, 10):
        rho = rand_dm_ginibre(2**n).data.todense()
        vec = StandardBasis.vec_from_rho(rho, 2 ** n)
        actual_rho = StandardBasis.rho_from_vec(vec, 2 ** n)
        assert_array_almost_equal(actual_rho, rho)

    # Test going from rho to vector.
    rho = np.array([[1., 1 + 2.j], [1 - 2.j, 3.]])
    desired_vec = np.array([1., 1., 2., 3.])
    actual_vec = StandardBasis.vec_from_rho(rho, 2)
    assert_array_almost_equal(actual_vec, desired_vec)

    # Test different dimension
    rho = np.array([[0.25, 0.2 + 0.1j, 0.3],
                    [0.5 - 0.1j, 0.25, 0.23 + 0.1j],
                    [0.3, 0.23 - 0.1j, 0.5]])
    dim = 3
    vec = StandardBasis.vec_from_rho(rho, dim)
    desired = [0.25, 0.2, 0.1, 0.3, 0, 0.25, 0.23, 0.1, 0.5]
    assert_array_almost_equal(desired, vec)

    assert StandardBasis.numb_variables(dim) == dim**2


def test_overparameterization():
    r"""Tests for "PyQCodes.param.OverParam"."""
    # Test numb_variables
    assert OverParam.numb_variables(5, 3) == 2 * 5 * 3

    matrix_size = 2
    # Test for 100 examples.
    for _ in range(0, 100):
        # Full Rank
        rank = 2

        v = np.random.uniform(-1, 1, 2 * 2 * rank)
        actual_rho = OverParam.rho_from_vec(v, matrix_size)
        a = np.array([[complex(v[0], v[1]), complex(v[2], v[3])],
                      [complex(v[4], v[5]), complex(v[6], v[7])]])
        desired_rho = a.dot(np.conj(a).T) / np.trace(a.dot(np.conj(a).T))
        assert np.linalg.matrix_rank(desired_rho) == rank
        assert_array_almost_equal(actual_rho, desired_rho)

        # Smaller rank
        rank = 1
        v = np.random.uniform(-1, 1, 2 * 2 * rank)
        actual_rho = OverParam.rho_from_vec(v, matrix_size, rank)
        a = np.array([[complex(v[0], v[1])], [complex(v[2], v[3])]])
        desired_rho = a.dot(np.conj(a).T) / np.trace(a.dot(np.conj(a).T))
        assert np.linalg.matrix_rank(desired_rho) == rank
        assert_array_almost_equal(actual_rho, desired_rho)

    # Test bigger matrix.
    matrix_size = 4
    for _ in range(0, 100):
        # Test full rank matrix.
        rank = 2
        # Get random vector to test.
        v = np.random.uniform(-1, 1, 2 * matrix_size * rank)
        actual_rho = OverParam.rho_from_vec(v, matrix_size)
        # Desired answer.
        a = np.array([[complex(v[0], v[1]), complex(v[2], v[3])],
                      [complex(v[4], v[5]), complex(v[6], v[7])],
                      [complex(v[8], v[9]), complex(v[10], v[11])],
                      [complex(v[12], v[13]), complex(v[14], v[15])]])
        desired_rho = a.dot(np.conj(a).T) / np.trace(a.dot(np.conj(a).T))
        assert np.linalg.matrix_rank(desired_rho) == rank
        assert_array_almost_equal(actual_rho, desired_rho)

    # Test rho to vector.
    # Test full-rank example.
    rho = np.array([[0.5, 0.3 + 0.1j], [0.3 - 0.1j, 0.5]])
    eigs, evecs = np.linalg.eigh(rho)
    matrix_A = evecs.dot(np.diag(np.sqrt(eigs)))
    desired_vec = [np.real(matrix_A[0, 0]), np.imag(matrix_A[0, 0]), np.real(matrix_A[0, 1]),
                   np.imag(matrix_A[0, 1]), np.real(matrix_A[1, 0]), np.imag(matrix_A[1, 0]),
                   np.real(matrix_A[1, 1]), np.imag(matrix_A[1, 1])]
    actual = OverParam.vec_from_rho(rho, 2, rank=2)
    assert_array_almost_equal(desired_vec, actual)
    # Test conversion backwards
    actual = OverParam.rho_from_vec(actual, 2, rank=2)
    assert_array_almost_equal(rho, actual)

    # Test smaller rank of 2 of column size 4.
    matrix_size = 4
    rank = 2
    # Get random rank2 rho to test.
    v = np.random.uniform(-1, 1, 2 * matrix_size * rank)
    rho = OverParam.rho_from_vec(v, matrix_size, rank=2)
    # Get eigenvalues and truncate close-to zero eigenvalues
    eigs, evecs = np.linalg.eigh(rho)
    eigs[np.abs(eigs) < 1e-5] = 0.
    # Eigenvalues are ordered in a increasing manner, hence first two eigenvalues are zero.
    matrix_A = evecs[:, 2:].dot(np.diag(np.sqrt(eigs[2:])))
    desired_vec = [np.real(matrix_A[0, 0]), np.imag(matrix_A[0, 0]), np.real(matrix_A[0, 1]),
                   np.imag(matrix_A[0, 1]), np.real(matrix_A[1, 0]), np.imag(matrix_A[1, 0]),
                   np.real(matrix_A[1, 1]), np.imag(matrix_A[1, 1]), np.real(matrix_A[2, 0]),
                   np.imag(matrix_A[2, 0]), np.real(matrix_A[2, 1]), np.imag(matrix_A[2, 1]),
                   np.real(matrix_A[3, 0]), np.imag(matrix_A[3, 0]), np.real(matrix_A[3, 1]),
                   np.imag(matrix_A[3, 1])]
    actual = OverParam.vec_from_rho(rho, matrix_size, rank=2)
    assert_array_almost_equal(actual, desired_vec)

    # Test conversion back to rho
    actual_rho = OverParam.rho_from_vec(actual, matrix_size, rank=2)
    assert_array_almost_equal(rho, actual_rho)


def test_cholesky_parameterization_rho_from_vector():
    r"""Tests for 'PyQCodes.param.CholeskyParam'."""
    for _ in range(0, 1000):
        n = 1
        rank = 2
        v = np.random.uniform(-5., 5., 2**n * (2**n + 1) - 2**n)
        actual_rho = CholeskyParam.rho_from_vec(v, 2 ** n)
        a = np.array([[complex(v[0], 0.), 0.], [complex(v[2], v[3]), complex(v[1], 0.)]])
        desired_rho = a.dot(np.conj(a).T) / np.trace(a.dot(np.conj(a).T))
        assert np.abs(np.linalg.matrix_rank(desired_rho) - rank) < 1e-4
        assert_array_almost_equal(actual_rho, desired_rho)
        assert np.abs(np.trace(actual_rho) - 1.) < 1e-10
        assert np.min(np.linalg.eigvalsh(actual_rho)) >= -1e-10

        rank = 1
        v = np.random.uniform(-5., 5., rank * (rank + 1) + 2 * (2**n - rank) * rank - rank)
        actual_rho = CholeskyParam.rho_from_vec(v, 2 ** n, rank)
        a = np.array([[complex(v[0], 0.)], [complex(v[1], v[2])]])
        desired_rho = a.dot(np.conj(a).T) / np.trace(a.dot(np.conj(a).T))
        assert np.linalg.matrix_rank(desired_rho) == rank
        assert_array_almost_equal(actual_rho, desired_rho)
        assert_array_almost_equal(actual_rho, np.conj(actual_rho.T))
        assert np.abs(np.trace(actual_rho) - 1.) < 1e-10
        assert np.min(np.linalg.eigvalsh(actual_rho)) >= -1e-10

        n = 2
        rank = 3
        v = np.random.uniform(-5., 5., CholeskyParam.numb_variables(2 ** n, rank))
        actual_rho = CholeskyParam.rho_from_vec(v, 2 ** n, rank)
        a = np.array([[complex(v[0], 0), 0., 0.],
                      [complex(v[3], v[4]), complex(v[1], 0.), 0.],
                      [complex(v[5], v[6]), complex(v[7], v[8]), complex(v[2], 0.)],
                      [complex(v[9], v[10]), complex(v[11], v[12]), complex(v[13], v[14])]])
        desired_rho = a.dot(np.conj(a).T) / np.trace(a.dot(np.conj(a).T))
        assert np.linalg.matrix_rank(desired_rho) == rank
        assert_array_almost_equal(actual_rho, desired_rho)
        assert_array_almost_equal(actual_rho, np.conj(actual_rho.T))
        assert np.abs(np.trace(actual_rho) - 1.) < 1e-10
        assert np.min(np.linalg.eigvalsh(actual_rho)) >= -1e-10

        rank = 2
        v = np.random.uniform(-5., 5., CholeskyParam.numb_variables(2 ** n, rank))
        actual_rho = CholeskyParam.rho_from_vec(v, 2 ** n, rank)
        a = np.array([[complex(v[0], 0), 0.],
                      [complex(v[2], v[3]), complex(v[1], 0.)],
                      [complex(v[4], v[5]), complex(v[6], v[7])],
                      [complex(v[8], v[9]), complex(v[10], v[11])]])
        desired_rho = a.dot(np.conj(a).T) / np.trace(a.dot(np.conj(a).T))
        assert np.linalg.matrix_rank(desired_rho) == rank
        assert_array_almost_equal(actual_rho, desired_rho)
        assert_array_almost_equal(actual_rho, np.conj(actual_rho.T))
        assert np.abs(np.trace(actual_rho) - 1.) < 1e-10
        assert np.min(np.linalg.eigvalsh(actual_rho)) >= -1e-10

    # Test conversion from rho to vector to rho.
    rho = np.array([[0.25, 0.2 + 0.3j], [0.2 - 0.3j, 0.75]])
    vec = CholeskyParam.vec_from_rho(rho, 2, 2)
    actual_rho = CholeskyParam.rho_from_vec(vec, 2, 2)
    assert_array_almost_equal(rho, actual_rho)


def test_choleskly_parameterization_vector_from_rho():
    r"""Test converting cholesky parameterization to vector from rho."""
    rho = np.array([[.25, .2 + .3j], [.2 - .3j, .75]])
    desired = np.array([0.5, 0.479583, 0.4, -0.6])  # Test with wolframalpha.

    actual = CholeskyParam.vec_from_rho(rho, 2, 2)
    assert_array_almost_equal(actual, desired)

    actual = CholeskyParam.vec_from_rho(rho, 2, None)
    assert_array_almost_equal(actual, desired)

    # Test Number of variables
    assert CholeskyParam.numb_variables(2, 2) == len(desired)


def test_assertion_error_parameterization_abc():
    r"""Test that the ParameterizationABC raises not implemented error."""
    assert_raises(NotImplementedError, ParameterizationABC.numb_variables, 5, 10)
    assert_raises(NotImplementedError, ParameterizationABC.bounds, 5)
    assert_raises(NotImplementedError, ParameterizationABC.rho_from_vec, 5, 10)
    try:
        ParameterizationABC.vec_from_rho(5, 10)
        ParameterizationABC.random_vectors(5, 10)
    except NotImplementedError:
        raise AssertionError("No error should be attempted.")


def test_bounds_on_different_parameterizations():
    r"""Test the bounds on the variables in the different parameterizations."""
    for matrix_size in range(1, 5):
        for rank in range(1, matrix_size):
            # Overparam
            bounds = OverParam.bounds(matrix_size, rank)
            for x, y in bounds:
                assert np.abs(x + 1) < 1e-4
                assert np.abs(y - 1) < 1e-4

            # Cholesky Parameterization
            l_bnd = np.random.random_integers(1, 5)
            u_bnd = np.random.random_integers(l_bnd, 10)
            bounds = CholeskyParam.bounds(matrix_size, rank, l_bnd, u_bnd)
            # Test the first rank are inbetween zero and u_bnd
            for i in range(0, rank):
                assert np.abs(bounds[i][0]) < 1e-4
                assert np.abs(bounds[i][1] - u_bnd) < 1e-4
            for i in range(rank, len(bounds)):
                assert np.abs(l_bnd - bounds[i][0]) < 1e-4
                assert np.abs(bounds[i][1] - u_bnd) < 1e-4

        # Standard
        bounds = StandardBasis.bounds(matrix_size)
        for x, y in bounds:
            assert np.abs(x + 1) < 1e-4
            assert np.abs(y - 1) < 1e-4


def test_random_vectors_generation_parameterizations():
    r"""Test the random vectors generated from the parameterizations."""
    # Non acceptable distribution
    assert_raises(TypeError, OverParam.random_vectors, 5, 4, 'w.e.')
    assert_raises(TypeError, CholeskyParam.random_vectors, 5, 3, "w.sas")

    np.random.seed(1)
    for _ in range(0, 10):
        random_vec = OverParam.random_vectors(5, 4, "normal")
        numb_vars = OverParam.numb_variables(5, 4)
        desired = np.random.normal(0., 1., size=(numb_vars))
        assert np.all(np.abs(random_vec - desired))

        random_vec = OverParam.random_vectors(5, 4, "uniform")
        numb_vars = OverParam.numb_variables(5, 4)
        desired = np.random.uniform(-1, 1., size=(numb_vars))
        assert np.all(np.abs(random_vec - desired))

        random_vec = CholeskyParam.random_vectors(5, 4, "normal")
        assert np.all(random_vec[:4] <= 1.)

        random_vec = CholeskyParam.random_vectors(5, 4, "uniform")
        assert np.all(random_vec[:4] <= 1.)
