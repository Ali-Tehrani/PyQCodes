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
import pytest
from PyQCodes.find_codes import effective_channel_with_stabilizers


def test_effective_channel_method_with_two_cat_code():
    r"""
    Test effective channel method against shor's paper with two cat code.

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

    # Pauli Errors for Depolarizing Channel.
    for p in [0., 0.1, 0.2, 0.5, 0.7, 0.8, 0.81, 0.811, 0.83, 0.85, 0.9, 0.95]:
        prob_no_error = p
        prob_error = (1. - prob_no_error) / 3.

        k1 = np.array([[1., 0.], [0., 1.]], dtype=np.complex128) * np.sqrt(prob_no_error)
        k2 = np.array([[0., 1.], [1., 0.]], dtype=np.complex128) * np.sqrt(prob_error)
        k3 = np.array([[0., -complex(0., 1.)], [complex(0., 1.), 0.]], dtype=np.complex128) * np.sqrt(prob_error)
        k4 = np.array([[1., 0.], [0., -1.]], dtype=np.complex128) * np.sqrt(prob_error)

        kraus = [k1, k2, k3, k4]

        result = effective_channel_with_stabilizers(bin_rep, code_param, kraus, optimize="coherent")
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


if __name__ == "__main__":
    test_effective_channel_method_with_two_cat_code()
    pass
    # test_effective_channel_method_with_two_cat_code()
    # test_effective_channel_method_with_three_cat_code()
    # test_effective_channel_method_with_four_cat_code()
