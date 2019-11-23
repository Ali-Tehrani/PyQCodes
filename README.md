PyQCodes
========
<a href='https://docs.python.org/3.6/'><img src='https://img.shields.io/badge/python-3.6-blue.svg'></a>

PyQCodes provides a set of functions to investigate coherent information,
minimum fidelity, certain optimization-based codes and stabilizer codes.

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
* Optimization-Based Codes.  See [file for more info](PyQCodes/CODE_README.md).
    - Optimizes the average fidelity over Recover/Encoding operators.
    - Effective Channel Method of Stabilizer Codes.

Examples
--------
Consider the bit-flip channel <img src="/tex/60eeab20dca4be5e5498159d0777700c.svg?invert_in_darkmode&sanitize=true" align=middle width=128.22161384999998pt height=24.65753399999998pt/> acting on a density matrix <img src="/tex/6dec54c48a0438a5fcde6053bdb9d712.svg?invert_in_darkmode&sanitize=true" align=middle width=8.49888434999999pt height=14.15524440000002pt/>:

$$
\mathcal{N}(\rho) = (1 - p)\rho + p X \rho X.
$$
    
```python
import numpy as np
from PyQCodes.chan.channel import AnalyticQChan

p = 0.25  # Probability of error.
kraus = [(1- p) * np.eye(2), p * np.array([[0., 1.], [1., 0.]])]
dim = 2, 2  # Dimension of H_A and H_B, respectively.
qubits = [1, 1]  # Maps one qubit to one qubit.
channel = AnalyticQChan(kraus, qubits, dim[0], dim[1])
```

Bit-flip channel is a unital channel, ie <img src="/tex/c22b8de6e2c7b1fbfc54afac9d69fa10.svg?invert_in_darkmode&sanitize=true" align=middle width=67.64487344999999pt height=24.65753399999998pt/>. This can be seen by:

```python
rho = np.eye(2)
new_rho = channel.channel(rho, n=1)
print(new_rho)  # Should be identity.
channel.entropy()
```

The 2-shot coherent information or minimum fidelity of <img src="/tex/6e8fd6045ec7bd4ef7fb13d25d31f890.svg?invert_in_darkmode&sanitize=true" align=middle width=32.73642404999999pt height=26.76175259999998pt/> can be optimized.
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
    pytest -m "not slow -v
```

License
=======
Generally, PyQCode has MIT license unless the directory/file says otherwise.


Acknowledgements
=================
TODO


Contact Info
============
If any questions, feel free to open up an issue or email at "atehrani@uoguelph.ca"
