Quantum Codes
=============

This sections discusses two types of code and the functionalities.

- Stabilizer Codes
    - Obtain Code Space of Stabilizer Code.
    - Apply Encoding, Decoding Circuit to ProjectQ engine.
    - Apply the circuit for syndrome measurement of stabilizer.
    - Obtain the logical operators.
 
- Optimization Codes



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
from projectq.ops import XGate

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

# Decode 
dephasing.decoding_circuit(eng, register, deallocate_nqubits=True)
```


Optimization Codes
==================


Effective Channel Method
------------------------

QVECTOR
-------


Average Fidelity
----------------

Algorithm Info
==============
Binary Representation
---------------------


