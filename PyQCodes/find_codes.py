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
from scipy.optimize import differential_evolution

from PyQCodes.chan.channel import AnalyticQChan
from PyQCodes.device_adaptor import ProjectQDeviceAdaptor
from PyQCodes.codes import StabilizerCode

r"""
Contains functions for finding optimization-based quantum codes.
One can optimize the average fidelity or coherent information of the channel 
with respect to either the decoder, encoder, or have them both fixed.
"""

__all__ = ["effective_coherent_info", "QDeviceChannel"]


def effective_coherent_info(kraus_encod, kraus_chan, kraus_decods, objective="coherent",
                            sparse=False, param="over"):
    r"""
    Calculates the effective coherent information of a error-correcting procedure.

    Given a [n, k] quantum code, where the channel acts on k-qubits, the encoder maps k-qubits to
    n-qubits, and decoder must map n-qubits to k-qubits.

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
                                       sparse=False, options=None):
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
        List of krauss operators as numpy arrays that are scalar multiples of the pauli group and
        hence represent a Pauli Channel.
    optimize : str
        If optimize is "coherent" (default), then it optimizes the coherent information.
        If optimize is "fidelity", then it optimizes the minimum fidelity.
    sparse : bool
        If True, then pauli elements of stabilizer code are sparse.
    options : None or dict
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

    """
    if not isinstance(optimize, str):
        raise TypeError("Optimize should be a string.")
    if not (optimize == "coherent" or optimize == "fidelity"):
        raise TypeError("Optimize should either be coherent or fidelity.")
    if 2**code_param[1] != pauli_errors[0].shape[1]:
        raise TypeError("Number of Columns of Pauli error does not match 2**k.")

    # Parameters for optimization
    # TODO: add coherent information parameters.

    # Set up the objects, stabilizer code, pauli-channel error, respectively.
    stab = StabilizerCode(stabilizer, code_param[0], code_param[1])

    # Figure out the dimensions of channel later.
    error_chan = AnalyticQChan(pauli_errors, [1, 1], 2, 2, sparse=sparse)
    error_chan.krauss.update_kraus_operators(code_param[0] // code_param[1])

    # Get Kraus Operator for encoder.
    encoder = stab.encode_krauss_operators()

    # Get the kraus operators for the stabilizers that anti-commute with each kraus operator.
    kraus = stab.kraus_operators_correcting_errors(error_chan.krauss.nth_kraus_ops, sparse=sparse)
    # Multiply by the encoder to get the full approximate error-correcting.
    kraus = [x.dot(encoder) for x in kraus]

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
    return result


class QDeviceChannel():
    r"""
    Represent a quantum circuit as a function 'circuit_chan' using the ProjectQ Engines.

    """

    def __init__(self, eng, register, circuit_chan, real_device=False):
        r"""
        Construct the QDeviceChannel.

        Parameters
        ----------
        engine : BasicEngine
            Should be one of the following types from 'ProjectQ.cengines'. Should
            be of type "BasicEngine" from ProjectQ. See their documentation
            for more details,
            "https://projectq.readthedocs.io/en/latest/projectq.cengines.html".
        register : list
            List of qubits in the register.
        circuit(eng, register) : callable
            Should take as input the engine from 'ProjectQ.cengines' and the register.
        real_device : bool
            Boolean indicating whether it's being runned on a real quantum device.

        """
        self._channel = circuit_chan
        self._real_device = real_device
        self._device = ProjectQDeviceAdaptor(eng, register, real_device)

    @property
    def eng(self):
        return self._device._eng

    @property
    def register(self):
        return self._device._register

    @property
    def channel(self):
        return self._channel

    @property
    def numb_qubits(self):
        return self._device._numb_qubits

    def estimate_average_fidelity_channel(self, ntimes, epsilon, cheat=False):
        r"""
        Estimate the average fidelity of the quantum channel.

        Parameters
        ----------
        ntimes : int
            The number of trials to sample.
        epsilon : float
            The approximation to the unitary two design.
        cheat : bool
            TODO

        Returns
        -------
        float :
            The estimated average fidelity of the quantum circuit.

        """
        return self._device.estimate_average_fidelity(self.channel, ntimes, epsilon, cheat)

    def estimate_average_fidelity_error_code(self, encoder, decoder, ntimes, epsilon):
        r"""
        Estimate average fidelity of encoder, with channel, with decoder as fixed functions.

        Parameters
        ----------
        encoder(eng, register) : callable
            The encoding circuit as a callable function that takes input engine and its register.
        decoder(eng, register) : callable
            The decoding circuit as a callable function that takes input engine and its register.
        ntimes : int
            The number of trials to sample.
        epsilon : float
            The approximation to the unitary two design.

        Returns
        -------
        float :
            The estimated average fidelity of the decoder composed with the channel composed
            with the encoder.

        """
        # Represent the channel for the error correcting process.
        def total_channel(eng, register):
            encoder(eng, register)
            self.channel(eng, register)
            decoder(eng, register)

        return self._device.estimate_average_fidelity(total_channel, ntimes, epsilon)

    def optimize_encoder_decoder_average_fidelity(self, encoder, decoder, ntimes, epsilon,
                                                  numb_parameters, bounds, optimizer="diffev",
                                                  maxiter=10):
        r"""
        Optimize the average fidelity of a channel of the parameters of the encoder and decoder.

        The encoder and decoder are paramterized by a set of parameters that are bounded.

        Parameters
        ----------
        encoder(eng, register, parameters) : callable
            A callable function that represents the encoder acting on the engine and register
            which is dependent on the parameters.
        decoder(eng, register, parameters) : callable
            A callable function that represents the decoder acting on the engine and register
            which is dependent on the parameters.
        ntimes : int
            The number of trials to sample.
        epsilon : float
            The approximation to the unitary two design.
        numb_parameters : tuple
            Tuple (M, N), where M is the number of parameters of the encoder and N is the number
            of parameters of the decoder.
        bounds : list of tuples
            List of tuples (l^i_bnd, u^i_bnd) that bound first the parameters of the encoder,
            then second the parameters of the decoder.
        optimizer : str
            Which optimizer to choose.
            If "diffev", then it optimizes using differential evolution from Scipy.
        maxiter : int
            The maximum number of iterations. Default is 10.

        Returns
        -------
        results : dict
            Returns a dictionary with the following fields.
                "success" : Whether it was successful or not.
                "optimal_val" : The optimal average fidelity found.
                "params_encoder" : The optimal parameters of the encoder.
                "params_decoder" : The optimal parameters of the decoder.

        """
        nparams_encoder = numb_parameters[0]
        nparams_decoder = numb_parameters[1]
        assert len(bounds) == nparams_encoder + nparams_decoder, "The length of bounds should be " \
                                                                 "the number of parameters of " \
                                                                 "encoder and decoder."

        def total_channel(eng, register, parameters):
            encoder(eng, register, parameters[:nparams_encoder])
            self.channel(eng, register)
            decoder(eng, register, parameters[nparams_encoder:])

        def objective_func(parameters, eng, register):
            return self._device.estimate_average_fidelity(total_channel, ntimes, epsilon)

        if optimizer == "diffev":
            result = differential_evolution(objective_func, bounds=bounds, maxiter=maxiter,
                                            args=(self.eng, self.register))
        else:
            raise ValueError("Does not recognize optimizer: " + str(optimizer))

        output = {"success" : result["success"], "params_encoder": result["x"][:nparams_encoder],
                  "params_decoder": result["x"][nparams_encoder:], "optimal_val": result["fun"]}
        return output
