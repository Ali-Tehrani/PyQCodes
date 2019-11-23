PyQCodes
========
<a href='https://docs.python.org/3.6/'><img src='https://img.shields.io/badge/python-3.6-blue.svg'></a>

PyQCodes provides a set of functions to investigate coherent information, minimum fidelity, certain optimization-based codes and stabilizer codes.

Guide To Do
-----------
* Quantum Channels. See [file for more info](PyQCodes/chan/README.md).
    - Compute Channel, Adjoint of Channel.
    - Compute Complementary Channel and Adjoint of Complementary Channel.
    - Optimize Coherent Information and Minimum Fidelity.
    - Serial/Parallel concatenate two channels .
* Stabilizer Codes.  See [file for more info](PyQCodes/README.md).
    - Find Logical Operators.
    - Apply Encoding, Measurement, and Decoding Circuit.
* Optimization-Based Codes.  See [file for more info](PyQCodes/README.md).
    - Optimizes the average fidelity over Recover/Encoding operators. Based on [^1]
    - Effective Channel Method of Stabilizer Codes. Based on [^2] and [^3].

[^1] : Channel-Adapted Quantum Error Correction for the Amplitude Damping Channel.
[^2] : 
[^3] : 

Examples
--------
Consider the bit-flip channel $\mathcal{N} : \mathcal{L}(\mathcal{H_A}) \rightarrow \mathcal{L}(\mathcal{H_B})$ acting on a density matrix $\rho$:
$$
\mathcal{N}(\rho) = (1 - p)\rho + p X \rho X.
$$
    
```python
import numpy as np
from PyQCodes.chan.channel import AnalyticQChan

p = 0.25  # Probability of error.
kraus = [(1- p) * np.eye(2), p * np.array([[0., 1.], [1., 0.]])]
dim = 2, 2  # Dimension of H_A and H_B, respectively.
qubits = [1, 1]  # Maps one qubit/particle to one qubit/particle.
channel = AnalyticQChan(kraus, qubits, dim[0], dim[1])
```

Bit-flip channel is a unital channel, ie $\mathcal{N}(I) = I$. This can be seen by:

```python
rho = np.eye(2)
new_rho = channel.channel(rho, n=1)
print(new_rho)  # Should be identity.
print("Von Neumann entropy is: ", channel.entropy(new_rho))

# Complementary channel isn't unital.
print("Complementary channel at I: ", channel.entropy_exchange(rho, n=1))
```

The 2-shot coherent information or minimum fidelity of $\mathcal{N}^{\otimes 
2}$ can be optimized. Note that global optima is not guaranteed.

```python
result = channel.optimize_coherent(n=2, rank=4, maxiter=100, disp=True)
# result = channel.optimize_fidelity(n=2, maxiter=100,)
print("Is successful: ", result["success"])
print("Optimal rho", result["optimal_rho"])
print("Optimal Value", result["optimal_val"])
```

Installing
----------
Before, installing make sure that Python 3.6 or higher and Cython (>= 0.21) is installed.

First clone to a directory by running:
```bash
   git clone https://github.com/Ali-Tehrani/PyQCodes
```

Then going to the PyQCode directory run
```bash
   pip install -e ./ --user
```

Finally, it is recommended to run the tests to see if it installed correctly.

```bash
    pytest -m "not slow" -v
```

License
=======
Generally, PyQCode has MIT license unless the directory/file says otherwise.


Acknowledgements
=================
Funded by the Huawei Innovation Fund TODO.


Contact Info
============
If any questions, feel free to open up an issue or email at "atehrani@uoguelph.ca"
