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
from projectq.ops import H, Measure, All, R
from projectq.meta import Control

from PyQCodes.chan.channel import AnalyticQChan
from PyQCodes.codes import StabilizerCode

r"""
Contains functions for finding optimization-based quantum codes.
One can optimize the average fidelity or coherent information of the channel 
with respect to either the decoder, encoder, or have them both fixed.
"""

__all__ = ["effective_coherent_info", "optimize_both"]


def optimize_encoder(kraus_chan, kraus_decod, objective="coherent", sparse=False,
                     param_dens="over", param_decoder=""):
    pass


def optimize_both(kraus_chan, objective="coherent", sparse=False,
                     param_dens="over", param_decoder=""):
    pass


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
    # TODO: Add error if n is not a multiple of k.
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
        self._eng = eng
        self._register = register
        self._numb_qubits = len(register)
        self._channel = circuit_chan
        self._real_device = real_device

    @property
    def eng(self):
        return self._eng

    @property
    def register(self):
        return self._register

    @property
    def channel(self):
        return self._channel

    @property
    def numb_qubits(self):
        return self._numb_qubits

    def set_register_to_zero(self, cheat=False):
        if not self._real_device:
            if cheat:
                probability = [1.] + [0.] * (self.numb_qubits - 1)
                self.eng.backend.set_wavefunction(probability, self.register)
            else:
                self.eng.flush(deallocate_qubits=True)
                self._register = self.eng.allocate_qureg(self.numb_qubits)
                self.eng.flush()

    def approximate_unitary_two_design(self, epsilon):
        r"""
        Construct the circuit for the epsilon-approximate unitary two design.

        Parameters
        ----------
        epsilon : float
            The error for the approximate unitary two design.

        Returns
        -------
        list :
            The random phases used for the circuit. This can is only intended to be used for the
            inverse of the unitary two design circuit.

        References
        ----------
        - See 'doi:10.1063/1.4983266'.

        Notes
        -----
        Note that this algorithm assumes there exists two qubit gates inbetween all qubits.
        """
        phases = [0, 2. * np.pi / 3., 4. * np.pi / 3.]
        l = int(np.ceil(np.log2(1. / epsilon) / self.numb_qubits))
        numb_times = 2 * l + 1

        # Store the random phases used, in-order to compute the inverse of the unitary 2-design.
        random_phases_used = []
        for _ in range(0, numb_times):
            ith_phases_used = []

            # Get uniformly sampled phases.
            random_phases = np.random.choice(phases, size=(self.numb_qubits,))
            # Apply the single qubit phase gates in the diagonal Z-basis.
            for i in range(0, self.numb_qubits):
                ith_phases_used.append(random_phases[i])
                R(random_phases[i]) | self.register[i]

            for i in range(0, self.numb_qubits):
                with Control(self.eng, self.register[i]):
                    for j in range(0, self.numb_qubits):
                        if i != j:
                            # Get a random phase.
                            phase = np.random.choice([0, np.pi], size=1)
                            ith_phases_used.append(phase[0])

                            R(phase) | self.register[j]

            # Store the phases used in the '_'th iteration.
            random_phases_used.append(ith_phases_used)
            All(H) | self.register
            self.eng.flush()

        # Reverse the random phases.
        random_phases_used.reverse()
        return random_phases_used

    def apply_inverse_of_unitary_two_design(self, random_phases):
        r"""
        Applies the inverse U^{-1} of a unitary 2-design generated from random phases.

        Parameters
        ----------
        random_phases : list of lists

        Returns
        -------

        """
        for ith_phases in random_phases:
            All(H) | self.register

            # Go though the inverse gates.
            counter = 1
            for i in range(self.numb_qubits - 1, -1, -1):
                with Control(self.eng, self.register[i]):
                    for j in range(self.numb_qubits - 1, -1, -1):
                        if i != j:
                            R(ith_phases[-counter]) | self.register[j]
                            counter += 1

            # Apply the single qubit phase gates in the diagonal Z-basis.
            for i in range(self.numb_qubits - 1, -1, -1):
                R(-ith_phases[-counter]) | self.register[i]
                counter += 1

            self.eng.flush()

    def estimate_average_fidelity(self, ntimes, epsilon):
        r"""
        Estimate the average fidelity by epsilon-approximate unitary two design.

        Parameters
        ----------
        ntimes : int
            The number of trials to sample.

        epsilon : float
            The approximation to the unitary two design.

        Returns
        -------
        float :
            The average fidelity of the quantum circuit.

        """
        # Set engine to all zero.

        # Number of times it is zero.
        ntimes_zero = 0.
        for _ in range(0, ntimes):
            # Set register to zero.
            self.set_register_to_zero(cheat=True)

            print(self.eng.backend.cheat())

            # Apply the approximate unitary two design.
            random_phase = self.approximate_unitary_two_design(epsilon)

            # Apply the quantum channel
            self.channel(self.eng, self.register)

            # Apply the inverse of the approximate unitary two design.
            self.apply_inverse_of_unitary_two_design(random_phase)

            # Measure in computational basis
            All(Measure) | self.register
            self.eng.flush()

            result = np.array([int(x) for x in self.register], dtype=np.int)
            if np.all(result == 0.):
                ntimes_zero += 1
        self.eng.flush()
        return ntimes_zero / ntimes
