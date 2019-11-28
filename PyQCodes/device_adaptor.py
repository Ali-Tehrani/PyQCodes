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
from abc import ABC, abstractmethod
import numpy as np

from projectq.ops import H, Measure, All, R
from projectq.meta import Control


r"""
This file contains functions that use the ProjectQ engine.

Particular importance is the ProjectQDeviceAdaptor which allows the estimation of average 
fidelity via approximate unitary two design.
"""

__all__ = ["ProjectQDeviceAdaptor"]


class DeviceAdaptorABC(ABC):
    r"""Abstract base class for device adaptors."""
    def __init__(self):
        pass

    @abstractmethod
    def approximate_unitary_two_design(self):
        pass

    @abstractmethod
    def estimate_average_fidelity(self):
        pass

    def decomp_one_qubit_c_unitaries(self):
        # Project Q already implements this, so it is probably not useful at all.
        pass


class ProjectQDeviceAdaptor(DeviceAdaptorABC):
    r"""
    Class that contains functions dealing with ProjectQ engine.

    Attributes
    ----------
    eng : BasicEngine
        The engine for the ProjectQ.
    register : list
        List of all qubits in the register for the ProjectQ engine.
    numb_qubits : int
        The number of qubits.

    Methods
    -------
    estimate_average_fidelity(channel, ntimes, epsilon) : int
        Given a callable channel, estimate the average fidelity of that channel using
        'epsilon'-approximate unitary two design with ntimes number of estimations.

    """

    def __init__(self, eng, register, real_device=False):
        r"""
        Construct the ProjectQDeviceAdaptor.

        Parameters
        ----------
        eng : BasicEngine
            The engine for the ProjectQ.
        register : list
            List of all qubits in the register for the ProjectQ engine.
        real_device : bool
            True if using a real-device.

        """
        self._eng = eng
        self._register = register
        self._numb_qubits = len(register)
        self._real_device = real_device
        super(DeviceAdaptorABC).__init__()

    @property
    def eng(self):
        r"""Return the ProjectQ engine."""
        return self._eng

    @property
    def register(self):
        r"""Return the qubit register that holds all the qubits."""
        return self._register

    @property
    def numb_qubits(self):
        r"""Returns the number of qubits."""
        return self._numb_qubits

    def set_register_to_zero(self, cheat=False):
        r"""
        Initializes the register to all zeros.

        Parameters
        ----------
        cheat : bool
            If True, then it will use the function "backend.set_wavefunction" to set the probability
            of the state |00..00> to 1 and rest zero.

        """
        if not self._real_device:
            if cheat:
                probability = [1.] + [0.] * (2**self.numb_qubits - 1)
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

    def inverse_of_unitary_two_design(self, random_phases):
        r"""
        Applies the inverse U^{-1} of a unitary 2-design generated from random phases.

        Parameters
        ----------
        random_phases : list of lists

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

    def estimate_average_fidelity(self, channel, ntimes, epsilon, cheat=False):
        r"""
        Estimate the average fidelity by epsilon-approximate unitary two design.

        Parameters
        ----------
        channel : callable
            This is a function that takes engine and register and implements the noisy quantum
            channel (e.g. a gate).
        ntimes : int
            The number of trials to sample.
        epsilon : float
            The approximation to the unitary two design.
        cheat : bool
            TODO

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
            self.set_register_to_zero(cheat=cheat)

            # Apply the approximate unitary two design.
            random_phase = self.approximate_unitary_two_design(epsilon)

            # Apply the quantum channel
            channel(self.eng, self.register)

            # Apply the inverse of the approximate unitary two design.
            self.inverse_of_unitary_two_design(random_phase)

            # Measure in computational basis
            All(Measure) | self.register
            self.eng.flush()

            result = np.array([int(x) for x in self.register], dtype=np.int)
            if np.all(result == 0.):
                ntimes_zero += 1
        self.eng.flush()
        return ntimes_zero / ntimes
