r"""
The files "_optimize.py", "find_code.py": For Optimizing Average Fidelity
    over convex set of quantum channels.
Copyright (C) 2019 Alireza Tehrani

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from PyQCodes.optimization.find_codes import effective_channel_with_stabilizers, optimize_decoder_stabilizers


def test_effective_channel_method_with_two_cat_code():
    r"""Test effective channel method against shor's paper with two cat code.

    Shor's method showed for the cat-code and dephasing channel with probability
    of no error occuring as f.

    If f  < 0.8113, then coherent information is zero
    and if f >= 0.8113, then coherent information is positive.
    """
    # Cat-code maps |0> -> |0>^2 and |1> -> |1>^2
    stab_code = ["ZZ"]
    bin_rep = np.array([[0, 0, 1, 1]])
    n, k = 2, 1
    code_param = (n, k)

    # Pauli Errors for Deplorizing Channel
    for p in np.arange(0, 1., 0.01):
        prob_no_error = p
        prob_error = (1. - prob_no_error) / 3.

        k1 = np.array([[1., 0.], [0., 1.]], dtype=np.complex128) * np.sqrt(prob_no_error)
        k2 = np.array([[0., 1.], [1., 0.]], dtype=np.complex128) * np.sqrt(prob_error)
        k3 = np.array([[0., -complex(0., 1.)], [complex(0., 1.), 0.]], dtype=np.complex128) * np.sqrt(prob_error)
        k4 = np.array([[1., 0.], [0., -1.]], dtype=np.complex128) * np.sqrt(prob_error)

        kraus = [k1, k2, k3, k4]

        result = effective_channel_with_stabilizers(bin_rep, code_param, kraus, opti="coherent")

        if p <= 0.8113:
            assert result["optimal_val"] <= 1e-8
        else:
            assert result["optimal_val"] > 1e-8


def test_effective_channel_method_with_three_cat_code():
    r"""Test effective channel method against shor's paper with three cat code.

    Shor's method showed for the cat-code and dephasing channel with probability
    of no error occuring as f.

    If f  < 0.8099, then coherent information is zero
    and if f >= 0.8099, then coherent information is positive.
    """
    # Three Cat-code maps |0> -> |0>^3 and |1> -> |1>^3
    stab_code = ["ZZI", "ZIZ", "IZZ"]
    bin_rep = np.array([[0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 1, 1]])
    n, k = 3, 1
    code_param = (n, k)

    # Pauli Errors for Deplorizing Channel
    for p in np.arange(0.8097, 1., 0.0001):
        prob_no_error = p
        prob_error = (1. - prob_no_error) / 3.

        k1 = np.array([[1., 0.], [0., 1.]], dtype=np.complex128) * np.sqrt(prob_no_error)
        k2 = np.array([[0., 1.], [1., 0.]], dtype=np.complex128) * np.sqrt(prob_error)
        k3 = np.array([[0., -complex(0., 1.)], [complex(0., 1.), 0.]], dtype=np.complex128) * np.sqrt(prob_error)
        k4 = np.array([[1., 0.], [0., -1.]], dtype=np.complex128) * np.sqrt(prob_error)

        kraus = [k1, k2, k3, k4]

        result = effective_channel_with_stabilizers(bin_rep, code_param, kraus, opti="coherent")
        print('p ', p)
        print("result ", result["optimal_val"])
        print("")
        if p <= 0.8099:
            assert result["optimal_val"] <= 1e-8
        else:
            assert result["optimal_val"] > 1e-8


def test_effective_channel_method_with_four_cat_code():
    r"""Test effective channel method against shor's paper using four cat code.

    Shor's method showed for the cat-code and dephasing channel with probability
    of no error occuring as f.

    If f  < 0.8101, then coherent information is zero
    and if f >= 0.8101, then coherent information is positive.
    """
    # Four Cat-code maps |0> -> |0>^4 and |1> -> |1>^4
    stab_code = ["ZZII", "ZIZI", "IZZI", "ZZZZ"]
    bin_rep = np.array([[0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1]])
    n, k = 4, 1
    code_param = (n, k)

    # Pauli Errors for Deplorizing Channel
    for p in np.arange(0.8100, 1., 0.0001):
        prob_no_error = p
        prob_error = (1. - prob_no_error) / 3.

        k1 = np.array([[1., 0.], [0., 1.]], dtype=np.complex128) * np.sqrt(prob_no_error)
        k2 = np.array([[0., 1.], [1., 0.]], dtype=np.complex128) * np.sqrt(prob_error)
        k3 = np.array([[0., -complex(0., 1.)], [complex(0., 1.), 0.]], dtype=np.complex128) * np.sqrt(prob_error)
        k4 = np.array([[1., 0.], [0., -1.]], dtype=np.complex128) * np.sqrt(prob_error)

        kraus = [k1, k2, k3, k4]

        result = effective_channel_with_stabilizers(bin_rep, code_param, kraus, opti="coherent")
        print('p ', p)
        print("result ", result["optimal_val"])
        print("")
        if p <= 0.8101:
            assert result["optimal_val"] <= 1e-8
        else:
            assert result["optimal_val"] > 1e-8


def test_optimization_average_fidelity_on_identtiy_channel():
    r"""Test optimizing decoder on the identity channel."""

    # Five qubit code
    stab_code = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    n = 5
    k = 1
    bin_rep = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                        [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]])


    #  Identity hannel on a single qubit.
    # kraus = [np.eye(2) / 2.]
    #
    # result = optimize_decoder_stabilizers(bin_rep, (n, k), kraus, sparse=True)
    # desired = 1
    # # TODO: Add assertion of result here.

    # Test on random unitary operator.
    L = np.random.normal(0., 1., size=(2, 2))
    mat = L.dot(L.conj().T)
    _, unitary = np.linalg.eigh(mat)
    assert np.all(np.abs(unitary.conj().T - np.linalg.inv(unitary)) < 1e-5)
    kraus = [unitary]
    result = optimize_decoder_stabilizers(bin_rep, (n, k), kraus, sparse=True)
    desired = 1


if __name__ == "__main__":
    test_optimization_average_fidelity_on_identtiy_channel()
    # test_effective_channel_method_with_two_cat_code()
    # test_effective_channel_method_with_three_cat_code()
    # test_effective_channel_method_with_four_cat_code()
