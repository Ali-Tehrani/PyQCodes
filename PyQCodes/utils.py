import numpy as np
from scipy.linalg import sqrtm


def turn_complex_into_real(mat):
    r"""
    Trun Complex Matrix into 2n x 2n real matrix.
    Note the iegenvalues becomes doubled.
    Note you got t
    Args:
        mat:

    Returns:

    """
    real_mat = np.zeros((mat.shape[0] * 2, mat.shape[0] * 2))
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[0]):
            actual_i = i * 2
            actual_j = j * 2

            a = np.real(mat[i, j])
            b = np.imag(mat[i, j])
            real_mat[actual_i , actual_j] = a
            real_mat[actual_i + 1, actual_j] = b
            real_mat[actual_i + 1, actual_j + 1] = a
            real_mat[actual_i, actual_j + 1] = -b
    return real_mat


def decompose_matrix(basis, matrix):
    r"""
    Decomposes a matrix based on a matrix basis set.

    In general, matrix basis set is the set of pauli operators.

    Parameters
    ----------
    basis : list
        List composed of matrix basis.

    matrix : array
        The matrix being decomposed.

    Returns
    -------
    array :
        Coefficients of matrix ddecomposed into linear combination of matrix basis
        elements.
    """
    assert matrix.shape == basis[0].shape, \
        "Matrix shape should equal matrix basis shape"
    coeff_matrix = np.zeros((basis[0].size, len(basis)), dtype=np.complex128)
    for i, basis_mat in enumerate(basis):
        coeff_matrix[:, i] = np.ravel(basis_mat)
    vect_mat = np.ravel(matrix)
    coeffs = np.linalg.pinv(coeff_matrix).dot(vect_mat)
    return coeffs


def fidelity_density(rho1, rho2):
    r"""
    Definition of the Fidelity based on density states.

    Parameters
    ----------
    rho1: array
        Hermitian, trace-one matrix.

    rho2: array
        Hermitian, trace-one matrix.

    Returns
    -------
    float :
        Fidelity between two density states.
    """
    sqrt1 = sqrtm(rho1)
    out = sqrtm(sqrt1.dot(rho2).dot(sqrt1))
    return np.trace(out)**2
