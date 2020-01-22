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
import numpy as np
import pytest
import projectq
from projectq.ops import All, Measure, XGate

from PyQCodes.find_codes import effective_channel_with_stabilizers, QDeviceChannel
from PyQCodes.device_adaptor import ProjectQDeviceAdaptor


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
    for p in [0.,0.8, 0.81, 0.811, 0.83]:
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


@pytest.mark.slow
def test_effective_channel_method_with_three_cat_code():
    r"""Test effective channel method against shor's paper with three cat code.

    Shor's method showed for the cat-code and dephasing channel with probability
    of no error occuring as f.

    If f  < 0.8099, then coherent information is zero
    and if f >= 0.8099, then coherent information is positive.
    """
    # Three Cat-code maps |0> -> |0>^3 and |1> -> |1>^3
    stab_code = ["ZZI", "ZIZ"]
    bin_rep = np.array([[0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 0, 1]])
    n, k = 3, 1
    code_param = (n, k)

    # Pauli Errors for Deplorizing Channel
    for p in np.arange(0.809, 0.811, 0.01):
        prob_no_error = p
        prob_error = (1. - prob_no_error) / 3.

        k1 = np.array([[1., 0.], [0., 1.]], dtype=np.complex128) * np.sqrt(prob_no_error)
        k2 = np.array([[0., 1.], [1., 0.]], dtype=np.complex128) * np.sqrt(prob_error)
        k3 = np.array([[0., -complex(0., 1.)], [complex(0., 1.), 0.]], dtype=np.complex128) * np.sqrt(prob_error)
        k4 = np.array([[1., 0.], [0., -1.]], dtype=np.complex128) * np.sqrt(prob_error)

        kraus = [k1, k2, k3, k4]

        result = effective_channel_with_stabilizers(bin_rep, code_param, kraus, optimize="coherent")
        if p <= 0.8099:
            assert result["optimal_val"] <= 1e-8
        else:
            assert result["optimal_val"] > 1e-8


# This test takes too long for now comment it unless one wants it.
# @pytest.mark.slow
# def test_effective_channel_method_with_four_cat_code():
#     r"""Test effective channel method against shor's paper using four cat code.
#
#     Shor's method showed for the cat-code and dephasing channel with probability
#     of no error occuring as f.
#
#     If f  < 0.8101, then coherent information is zero
#     and if f >= 0.8101, then coherent information is positive.
#     """
#     # Four Cat-code maps |0> -> |0>^4 and |1> -> |1>^4
#     stab_code = ["ZZII", "IZZI", "ZZZZ"]
#     bin_rep = np.array([[0, 0, 0, 0, 1, 1, 0, 0],
#                         [0, 0, 0, 0, 0, 1, 1, 0],
#                         [0, 0, 0, 0, 1, 1, 1, 1]])
#     n, k = 4, 1
#     code_param = (n, k)
#
#     # Pauli Errors for Deplorizing Channel
#     for p in np.arange(0.8100, 0.83, 0.01):
#         prob_no_error = p
#         prob_error = (1. - prob_no_error) / 3.
#
#         k1 = np.array([[1., 0.], [0., 1.]], dtype=np.complex128) * np.sqrt(prob_no_error)
#         k2 = np.array([[0., 1.], [1., 0.]], dtype=np.complex128) * np.sqrt(prob_error)
#         k3 = np.array([[0., -complex(0., 1.)], [complex(0., 1.), 0.]], dtype=np.complex128) * np.sqrt(prob_error)
#         k4 = np.array([[1., 0.], [0., -1.]], dtype=np.complex128) * np.sqrt(prob_error)
#
#         kraus = [k1, k2, k3, k4]
#
#         result = effective_channel_with_stabilizers(bin_rep, code_param, kraus,
#                                                     optimize="coherent", sparse=False)
#
#         if p <= 0.8101:
#             assert result["optimal_val"] <= 1e-8
#         else:
#             assert result["optimal_val"] > 1e-8


def test_inverse_of_unitary_two_design():
    r"""Test inverse of unitary two design with the unitary two design is the identity."""
    eng = projectq.MainEngine()
    register = eng.allocate_qureg(4)
    eng.flush()  # Need to flush it in order to set_wavefunction

    def identity(engine):
        pass

    chan_dev = ProjectQDeviceAdaptor(eng, register)

    for index in range(0, 2 ** 4):
        for _ in range(0, 25):
            # Set the wave-function to match the basis state.
            zero = [0.] * (2**4)
            zero[index] = 1  # Turn Probability is one on the that basis state.
            eng.backend.set_wavefunction(zero, register)
            eng.flush()

            epsilon = 1
            phases = chan_dev.approximate_unitary_two_design(epsilon)
            chan_dev.inverse_of_unitary_two_design(phases)

            wave_function_after = np.array(eng.backend.cheat()[1])
            assert np.all(np.abs(wave_function_after - np.array(zero)) < 1e-3)
    All(Measure) | register


def test_average_fidelity_on_the_identity_channel():
    r"""Test that the average fidelity on the identity channel."""
    qubits = 4
    dim = 2**qubits

    true_answer = (np.abs(np.trace(np.eye(dim))**2) + dim) / (dim**2 + dim)

    # Construct the channel
    eng = projectq.MainEngine()
    register = eng.allocate_qureg(qubits)
    eng.flush()  # Need to flush it in order to set_wavefunction

    def identity(engine, register):
        pass

    chan_dev = QDeviceChannel(eng, register, identity)
    fidelity = chan_dev.estimate_average_fidelity_channel(100, 0.1)
    assert np.abs(fidelity - true_answer) < 1e-4


def test_average_fidelity_on_bit_flip_chanel():
    r"""Test the average fidelity of the bit flip map."""
    qubits = 1
    dim = 2**qubits
    for i, prob_error in enumerate([1., 0.5, 0.25]):  # Test different probabilities
        # Obtain the actual true answer
        kraus = [np.sqrt(1 - prob_error) * np.eye(dim),
                 np.sqrt(prob_error) * np.array([[0., 1.],
                                                 [1., 0.]])]
        true_answer = (np.abs(np.trace(kraus[0]))**2 + np.abs(np.trace(kraus[1]))**2 + dim) /\
                      (dim**2 + dim)

        # Construct the channel and perform average fidelity estimation.
        eng = projectq.MainEngine()
        register = eng.allocate_qureg(qubits)
        eng.flush()  # Need to flush it in order to set_wavefunction

        def bit_flip(engine, register):
            rando = np.random.random()  # Get uniform random number from zero to one..
            if rando < prob_error:  # If it is less than probability of error.
                XGate() | register
            engine.flush()

        chan_dev = QDeviceChannel(eng, register, bit_flip)

        fidelity = chan_dev.estimate_average_fidelity_channel(500, 0.001, cheat=bool(i % 2))
        assert np.abs(fidelity - true_answer) < 1e-1
