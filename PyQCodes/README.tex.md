Quantum Codes
=============

This sections discusses two types of code and the functionalities.

- Stabilizer Codes
    - Obtain Code Space of Stabilizer Code.
    - Apply Encoding, Decoding Circuit to ProjectQ engine.
    - Apply the circuit for syndrome measurement of stabilizer.
    - Obtain the logical operators.
 
- Optimization Codes
    - Effective Channel Method.
    - Approximate Average Fidelity/QVECTOR
    - Channel Adaptive Codes (Work In Progress)


Stabilizer Codes
----------------
Consider the dephasing stabilizer code specified by stabilizers $\{XXI, XIX\}$.
This has a binary representation as

$$
\begin{bmatrix}
    1& 1& 0  &0& 0& 0\\
    1& 0& 1  &0& 0& 0 
\end{bmatrix}
$$

The dephasing code is a (3, 1) code which can be created as follows:
```python
import numpy as np
from PyQCodes.codes import StabilizerCode

# Created by adding the pauli string to stabilizer.
code = ["XXI", "XIX"]
dephasing = StabilizerCode(code, 3, 1)

# Or Can be created using binary representation.
binary = np.array([[1, 1, 0, 0, 0, 0],
                   [1, 0, 1, 0, 0, 0]])
dephasing = StabilizerCode(code, 3, 1)
```

The basis of the code space of the stabilizer code is the common eigenspace 
of eigenvalue one. It can be obtained as follows:
```python
# The column vectors are the basis of the code space.
kraus = dephasing.encode_krauss_operators(sparse=False)
```

The StabilizerCode object of a (n, k)-code can also work for the ProjectQ
circuits. 
It can encode any k-qubit state using the code.
There are a total of k X-logical operators and k Z-logical operators.
It can decode back from n-qubit to k-qubit states.
```python
from projectq.cengines import MainEngine
from projectq.ops import All, Measure

# Create and allocate three qubits.
eng = MainEngine()
register = eng.allocate_qureg(3)

# Apply the encoding circuit on the 1-qubit state |0>
dephasing.encoding_circuit(eng, register, [0])

# There are 1 X-logical operators and 1 Z-logical operators.
dephasing.logical_circuit(eng, register, "X0")
# dephasing.logical_circuit(eng, register, "Z0")

# Apply decoding circuit to the 1-qubit state and delocate the last two qubits.
dephasing.decoding_circuit(eng, register, deallocate_nqubits=True)

Measure | register[0]

# Since we applied the X-logical operator, it should be |1>.
print("Should be one: ", int(register[0]))
```

A quantum error correction can be modeled.
```python
from projectq.ops import XGate, All, Measure

# Bit-flip map with probability of 0.1 as error.
def channel(eng, register):
    # Go through each qubit in the register.
    for i in range(0, len(register)):
        # Apply bit-flip map if probability < 0.1.
        prob = np.random.random()
        if prob < 0.1:
            XGate() | register[i]
         
# Create and allocate three qubits. Always flush after allocating.
eng = MainEngine()
register = eng.allocate_qureg(3)
eng.flush()

# Apply the encoding circuit on the 1-qubit state |0>
dephasing.encoding_circuit(eng, register, [0])

# Apply the bit-flip error channel.
channel(eng, register)

# Go Through and Measure each stabilizer.
for stab in dephasing:
    measurement = dephasing.single_syndrome_measurement(eng, register, stab)
    
    # Error was found.
    if measurement == 1:
        # Apply the stabilizer to reverse the error.
        dephasing.apply_stabilizer_circuit(eng, register, stab)
        break

# Decode 
dephasing.decoding_circuit(eng, register, deallocate_nqubits=True)

# Deallocate the register so it exits properly.
All(Measure) | register
```


Optimization Codes
==================

PyQCodes can perform certain optimization-based codes based on either
coherent information or fidelity.

Effective Channel Method
------------------------
The effective channel method optimizes either fidelity or coherent 
information of the effective channel, the encoding, channel and recovery 
operator.

For example, first define the kraus operators of the depolarizing channel 
with probability of no error being 0.1.
```python
import numpy as np

prob_no_error = 0.1
prob_error = (1. - prob_no_error) / 3.

k1 = np.array([[1., 0.], [0., 1.]]) * np.sqrt(prob_no_error)
k2 = np.array([[0., 1.], [1., 0.]]) * np.sqrt(prob_error)
k3 = np.array([[0., -complex(0., 1.)], [complex(0., 1.), 0.]]) * np.sqrt(prob_error)
k4 = np.array([[1., 0.], [0., -1.]]) * np.sqrt(prob_error)

kraus = [k1, k2, k3, k4]
```

The second step is to define the stabilizer code. Here, the
(2, 1) cat-code will be used which has stabilizers "ZZ" and binary 
representation $\begin{bmatrix}0 & 0 & 1 & 1 \end{bmatrix}$.

```python
from PyQCodes.codes import StabilizerCode

stab_code = ["ZZ"]
# Binary Representation of Cat Code.
bin_rep = np.array([[0, 0, 1, 1]])
code_param = (2, 1)
```

It will first create the effective channel, which will then be optimized 
using the coherent information.

```python
from PyQCodes.find_codes import effective_channel_with_stabilizers
result = effective_channel_with_stabilizers(bin_rep, code_param, kraus, optimize="coherent")

print("Optimal Value: ", result["optimal_val"])
```


Approximate Average Fidelity / QVECTOR
---------------------------------------
Consider the bit-flip map applied on one qubit with probability of error 0.1.

```python
import numpy as np
import projectq
from projectq.ops import XGate

# Construct the engine and qubits.
eng = projectq.MainEngine()
register = eng.allocate_qureg(1)
eng.flush()  # Need to flush it in order to set_wavefunction

def bit_flip(engine, register):
    rando = np.random.random()  # Get uniform random number from zero to one..
    if rando < 0.1:  # If it is less than probability of error.
        XGate() | register  # Apply the error.
    engine.flush()
```

The average fidelity can be approximated using approximate unitary 2-design.
```python
from PyQCodes.find_codes import QDeviceChannel

# Construct the Quantum Device Channel
chan_dev = QDeviceChannel(eng, register, bit_flip)
# Apply the algorithm using 500 trials and epsilon approximation of 0.001.
fidelity = chan_dev.estimate_average_fidelity_channel(500, 0.001)
print("The Approximate Fidelity is: ", fidelity)
```

TODO: Insert example of how to do QVECTOR


Channel Adaptive Codes (Work in Progress)
-----------------------------------------

