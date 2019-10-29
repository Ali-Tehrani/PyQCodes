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

from PyQCodes.channel import AnalyticQChan
from PyQCodes.codes import StabilizerCode
from PyQCodes.optimization._optimize import optimize_average_entang_fid

r"""
Contains functions for finding optimization-based quantum codes.
One can optimize the average fidelity or coherent information of the channel 
with respect to either the decoder, encoder, or have them both fixed.
"""

__all__ = ["effective_coherent_info", "optimize_decoder", "optimize_encoder",
           "optimize_both"]


def optimize_decoder_stabilizercode():
    # TODO: Do optimize decoder stabilizer code.
    pass


def optimize_decoder(encoder, kraus_chan, numb_qubits, dim_in, dim_out, objective="coherent",
                     sparse=False, param_dens="over", param_decoder=""):
    #TODO:

    # Set up stabilizer code, pauli-channel error and update the channel to match encoder.
    error_chan = AnalyticQChan(kraus_chan, numb_qubits, dim_in, dim_out)
    error_chan.krauss.update_kraus_operators(code_param[0] // code_param[1])

    # Kraus operators for channel composed with encoder.
    kraus_chan = [x.dot(encoder) for x in error_chan.nth_kraus_operators]

    numb_qubits = [code_param[1], code_param[0]]
    dim_in, dim_out = 2, 2
    result = optimize_average_entang_fid(kraus_chan, numb_qubits, dim_in, dim_out)
    return result


def optimize_decoder_stabilizers(stabilizer, code_param, kraus_chan, sparse=False,
                                 param_dens="over"):
    r"""
    Optimizes the minimum fidelity with respect to set of decoder.

    Notes
    -----
    - Assumes that all dimensions of single-particle hilbert space (input and output of channel)
        is 2.

    """
    # TODO: Add sparse option here.
    # Set up stabilizer code, pauli-channel error and update the channel to match encoder.
    stab = StabilizerCode(stabilizer, code_param[0], code_param[1], None, None)
    error_chan = AnalyticQChan(kraus_chan, [1, 1], 2, 2, sparse=sparse)
    error_chan.krauss.update_kraus_operators(code_param[0] // code_param[1])

    # Get Kraus Operator for encoder.
    encoder = stab.encode_krauss_operators(sparse=sparse)

    # Kraus operators for channel composed with encoder.
    if sparse:
        kraus_chan = [x.tocsr().dot(encoder) for x in error_chan.nth_kraus_operators]
    else:
        kraus_chan = [x.dot(encoder) for x in error_chan.nth_kraus_operators]

    numb_qubits = [code_param[1], code_param[0]]
    dim_in, dim_out = 2, 2
    result = optimize_average_entang_fid(kraus_chan, numb_qubits, dim_in, dim_out)
    return result


def optimize_encoder(kraus_chan, kraus_decod, objective="coherent", sparse=False,
                     param_dens="over", param_decoder=""):
    pass


def optimize_both(kraus_chan, objective="coherent", sparse=False,
                     param_dens="over", param_decoder=""):
    pass


def effective_coherent_info(kraus_encods, kraus_chan, kraus_decods, objective="coherent",
                            sparse=False, param="over"):
    r"""
    Calculates the effective coherent information of a error-correcting procedure.

    Given a [n, k] quantum code, where the channel acts on k-qubits,
    the encoder maps k-qubits to n-qubits, and decoder must map n-qubits to k-qubits.

    Parameters
    ----------
    kraus_encod : list or array
        List of numpy arrays corresponding to the kraus operators of the encoder or
        choi-matrix can be provided, where it is then converted to kraus representation.

    kraus_chan : list or array
        List of numpy arrays corresponding to the kraus operators of the channel or
        choi-matrix can be provided, where it is then converted to kraus representation.

    kraus_decod : list or array
        List of numpy arrays corresponding to the kraus operators of the decoder or
        choi-matrix can be provided, where it is then converted to kraus representation.

    objective ; string
        If "coherent", calculate and optimize the coherent information of the channel.
        If "fidelity", calculate the average fidelity of the channel.

    sparse : bool
        True, if one wants to use sparse krauss operators. Default option is False.

    param : string
        If "over", uses the Over-Parameterization for parameterizing density matrices.
        if "choleskly" uses the Choleskly Parameterization instead.
        Default and recommended option is "over". Use "choleskly" for high-dimensions.

    Returns
    -------
    dict :
        Returns Dictionary of the objective function, success of optimization and .

    """
    pass


def _optimization_coherent_information_parameters(dict_params):
    pass


def effective_channel_with_stabilizers(stabilizer, code_param, pauli_errors, optimize="coherent",
                                       sparse=False, opti_params={"cd"}):
    r"""
    Calculate the effective channel with respect to a set of pauli-errors and stabilizer elements.

    Parameters
    ----------
    stabilizer : list
        List of strings representing stabilizer elements.

    code_param : tuple
        A tuple (n, k) where n is the number of encoded qubits and k is the number of logical
        qubits.

    pauli_errors : list
        List of krauss operators as numpy arrays that are elements of pauli group.

    optimize : str
        If optimize is "coherent" (default), then it optimizes the coherent information.
        If optimize is "fidelity", then it optimizes the minimum fidelity.

    sparse : bool
        If True, then pauli elements of stabilizer code are sparse.

    opti_params : None or dict
        Dictionary of parameters for 'AnalyticQCodes.optimize_coherent' and
        'AnalyticQCodes.optimize_fidelity' optimizations procedures. Should have parameters,
        'param' : str
            Parameterization of density matrix, default is 'overparam'. Can also be 'cholesky.'
            Can also provide own's own parameterization by being a subclass of ParameterizationABC.
        'lipschitz': int
            Integer of number of samplers to be used for lipschitz sampler. Default is 50
        'use_pool' : int
            The number of pool proccesses to be used by the multiprocessor library. Default is 3.
        'maxiter' : int
            Maximum number of iterations. Default is 500.
        'samples' : list, optional
            List of vectors that satisfy the parameterization from "param", that are served as
            initial guesses for the optimization procedure. Optional, unless 'lipschitz' is zero.

    Returns
    -------
    float :
        The coherent information or minimum fidelity of the channel with error-correction.
    """
    if not isinstance(optimize, str):
        raise TypeError("Optimize should be a string.")
    if not (optimize == "coherent" or optimize == "fidelity"):
        raise TypeError("Optimize should either be coherent or fidelity.")
    if 2**code_param[1] != pauli_errors[0].shape[1]:
        raise TypeError("Number of Columns of Pauli error does not match 2**k.")

    # Parameters for optimization
    # TODO: Add error if n is not a multiple of k.
    # TODO: add coherent information parameters.

    # Set up the objects, stabilizer code, pauli-channel error, respectively.
    stab = StabilizerCode(stabilizer, code_param[0], code_param[1], None, None)
    # Figure out the dimensions of channel later.
    error_chan = AnalyticQChan(pauli_errors, [1, 1], 2, 2, sparse=sparse)
    error_chan.krauss.update_kraus_operators(code_param[0] // code_param[1])

    # Get Kraus Operator for encoder.
    encoder = stab.encode_krauss_operators()

    # Find stabilizer that anti-commute with each krauss operator.
    kraus = []
    pauli_stab = StabilizerCode.binary_rep_to_pauli_mat(stab.stab_bin_rep, sparse)

    for i, error in enumerate(error_chan.krauss.nth_kraus_ops):
        found_decoder = False
        # Go through each stabilizer element.
        for p in pauli_stab:
            p_error = p.dot(error)
            error_p = error.dot(p)

            # Check if they are anti-commuting.
            if np.all(np.abs(error_p + p_error) < 1e-5):
                # Append total kraus operators for entire error-correcting procedure.
                kraus.append(p.dot(error).dot(encoder))
                found_decoder = True  # Error is correctable.
                break

        # If the error commutes with all stabilizer elements.
        if not found_decoder:
            kraus.append(error.dot(encoder))

    # Construct new kraus operators.
    total_chan = AnalyticQChan(kraus, [code_param[1], code_param[0]], 2, 2, sparse=True)

    # Solve the objective function
    if optimize == "coherent":
        result = total_chan.optimize_coherent(n=1, rank=2,
                                              optimizer="slsqp", param="overparam",
                                              lipschitz=25, use_pool=3, maxiter=250)
    else:
        result = total_chan.optimize_fidelity(n=1, optimizer="slsqp", param="overparam",
                                              lipschitz=25, use_pool=3, maxiter=250)

    # Check for errors/success

    # Return the output.
    return result
