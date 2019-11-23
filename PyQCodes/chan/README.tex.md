Quantum Channels
==================

Overview
--------

This module contains the AnalyticQChan object which models the quantum channels as a kraus operator (can also do Choi matrices but is not recommended).

It has the following properties and methods that can be of used.

- Compute the channel, complementary channel (called entropy exchange here), the adjoint of a channel and adjoint of complementary channel.
- Can add (serial concatenate), multiply (parallel concatenate) differentchannels together.
- Compute the Von Neumann entropy and fidelity of a state wrt to a channel.
- Maximizes the coherent Information of a channel.
- Minimizes the minimum fidelity of a channel.
- Model kraus operators as sparse matrices.
- Calculate the average fidelity of a channel with respect to an ensemble.

See the code [documentation](channel.py) of AnalyticQChan for more information.

Examples
========
Setting up Dephrasure Channel
-----------------------------
Consider the dephrasure channel $\mathcal{N} : \mathcal{L}(\mathcal{H}_2) 
\rightarrow \mathcal{L}(\mathcal{H}_3)$ that maps a qubit to a qutrit. It was
 introduced in the paper [^1]. It has the following action on qubit density 
 matrix $\rho$:
 
 $$
 \mathcal{N}(\rho) = (1 - p)(1 - q) I \rho I + p(1 - q)Z \rho Z + q |e\rangle
  \langle e|.
 $$
 
 It has four kraus operators, the first two relating to I and Z and the last 
 two relating to erasure $|e\rangle\langle e|$.
 
 ```python
import numpy as np
from PyQCodes.chan.channel import AnalyticQChan
 
# Set up kraus operators
p, q = 0.1, 0.2
krauss_1 = np.array([[1., 0.], [0., 1.], [0., 0.]])
krauss_2 = np.array([[1., 0.], [0., -1.], [0., 0.]])
krauss_3 = np.array([[0., 0.], [0., 0.], [1., 0.]])
krauss_4 = np.array([[0., 0.], [0., 0.], [0., 1.]])
krauss_ops = [krauss_1 * np.sqrt((1 - p) * (1 - q)),
              krauss_2 * np.sqrt((1 - q) * p),
              np.sqrt(q) * krauss_3,
              np.sqrt(q) * krauss_4]

# Set up parameters
numb_qubits = [1, 1]  # It maps one qubit to one qubit.
dim_in = 2  # Dimension of H_2 (qubit).
dim_out = 3  # Dimension of output H_3 (qutrit).

channel = AnalyticQChan(krauss_ops, numb_qubits, dim_in, dim_out, sparse=False)
```

The sparse keyword turns the kraus operators into sparse matrices. It is 
recommended to use this when the number of kraus operators are large and sparse.


General Properties
------------------
The AnalyticQChan object can compute the channel $\mathcal{N}(\rho)$, the 
complementary channel $\mathcal{N}^c(\rho)$, and adjoint of each of these 
matrices. And it is able to compute channel $\mathcal{N}^{\otimes n}$ tensored n times (although it grows fast!).

```python
rho = np.array([[0.5, 0.2 + 0.3j], [0.2 - 0.3j, 0.5]])
chan_rho = channel.channel(rho, n=1)  # Evaluate the channel at rho.
comp_rho = channel.entropy_exchange(rho, n=1)  # Evaluate complementary channel.

rho = np.array([[0.25, 0., 0.], [0., 0.25, 0], [0., 0., 0.5]])
adjoint_chan = channel.channel(rho, n=1, adjoint=True)  # Evaluate adjoint channel.
adjoint_comp = channel.entropy_exchange(rho, n=1, adjoint=True)  # Evaluate adjoint of complementary channel

# Similarly compute the channel tensored two times at a maximally mixed state.
rho = np.diag([0.25, 0.25, 0.25, 0.25])
chan_rho = channel.channel(rho, n=2)
```

Two channels $\mathcal{N}_1$ and $\mathcal{N}_2$ can be used to create a third channel $\mathcal{N}_2 \circ \mathcal{N}_1$ via serial concatenation.
```python
chan1 = AnalyticQChan(kraus1, [1, 1], 2, 3)
chan2 = AnalyticQChan(kraus2, [1, 1], 2, 3)
new_channel = chan2 + chan1
```

Similarly, two channel can create a third tensored channel $\mathcal{N}_1 \otimes \mathcal{N}_2$ via parallel concatenation.
```python
new_channel = chan2 * chan1
# Note that the new channel will always assume it will map one qubit to one qubit.
```

Note that the complementary channel $\mathcal{N}^c$ maps a density matrix to 
the environment system, 
here it is defined as the matrix with ith jth entries as $Tr(A_i \rho A_j^\dagger).$  

Coherent Information
--------------------
The coherent information of a channel $I_c(\mathcal{N})$ is defined to be

$$
    I_c(\mathcal{N}) = \max_{\rho} S(\mathcal{N}(\rho)) - S(\mathcal{N}^c(\rho)).
$$

Maximization is done by parameterizing rank-k density matrices (See 
[below](#parameterization)) using the OverParameterization. Using the 
dephrasure example above.

```python
results = channel.optimize_coherent(n=2, rank=4, optimizer="diffev", 
                                    param="overparam", lipschitz=50, 
                                    use_pool=3, maxiter=100, disp=True)
print("Is successful: ", results["success"])
print("Optimal rho", results["optimal_rho"])
print("Optimal Value", results["optimal_val"])
```
This will optimize coherent information of $\mathcal{N}^{\otimes 2}$ over rank four (the highest rank) density matrices. It is generally recommended to
optimize over the highest rank. The optimization procedure will use "differential_evolution" from Scipy. It will generate 50 initial population 
sizes using the lipschitz sampler. It will execute 3 processes from the 
multiprocessing library to run it faster and will have a maximum iteration of 100.
The disp keyword will print the results as the optimization procedure 
progresses. "overparam" means it will use OverParameterization.

See [documentation](channel.py) of the method 'optimize_coherent' 
for more info.

It is highly recommended to use lipschitz sampler and use_pool and to optimize over the highest rank.
Furthermore, using lipschitz sampler and optimizer="slsqp" generally outperforms "differential_evolution".
For large n, you may need to reconstruct the AnalyticQChan using the sparse 
keyword as True. This does not guarantee to find the global optima, but it 
will find a local maxima.


Minimum Fidelity
----------------
The fidelity of a quantum state $\rho$ with respect to a channel 
$\mathcal{N}$ is:

$$
 F(\rho, \mathcal{N}(\rho)) = Tr\bigg(\sqrt{\sqrt{\rho}\mathcal{N}(\rho)\sqrt{\rho}}\bigg)
$$
The minimum fidelity is the minimum of the fidelity of a channel over rank one density matrices.

Here, the minimum fidelity over the dephrasure channel $\mathcal{N}^{\otimes 2}$ is applied here.

```python
results = channel.optimize_fidelity(n=2, optimizer="diffev", lipschitz=50, 
                                    use_pool=3, maxiter=100, disp=True)
print("Is successful: ", results["success"])
print("Optimal rho", results["optimal_rho"])
print("Optimal Value", results["optimal_val"])
```

The same parameters as optimizing coherent information is applied here.

See [documentation](channel.py) of the method 'optimize_fidelity' for more info.


Algorithm Information
=====================

Parameterization
----------------
This section is concerned with how the density matrices are parameterization.
These are located in the [file](param.py).

There are three options:

Parameterization | Advantages
------------ | -------------
OverParameterization | More Variables, seems to outperform Choleskly.
CholesklyParameterization | Less Variables, seems less accurate.
Own Parameterization | Freedom of choice but need to code.


OverParameterization and Choleskly Parameterization depend on the following 
fact that all rank-k positive semidefinite matrices $A$ of size $n \times n$ can
 be written as $L^\dagger L$ where $L$ is a size $k \times n$ complex matrix.
 
The OverParameterization method models $L$ as a $2 kn$ vector, where the two
is because every complex number is two real-vectors.
  
The CholesklyParameterization method models $L$ as a lower-triangular 
matrix and is modelled by a  $k(k+ 1) + 2(n - k)k - k$ real-vector.

Each real-variable in the matrix $L$ is bounded from -1 to 1. Furthermore,
to insure trace one, the matrix $A$ is $L^\dagger L / Tr(L^\dagger L)$.

If own wants to use their own parameterization this can be done by importing 
the abstract base class ParameterizationABC and implementing the three methods
'numb_variables', 'bounds(nsize, rank)' and 'rho_from_vec(vec, nsize)'.

Consider the following parameterization found in paper [^1],   
$$
    \rho = \delta|0\rangle \langle 0|^{\otimes nsize} + (1 - \delta) |1\rangle\langle 1|^{\otimes nsize}
$$
It has one real variables and the delta is bounded from 0 to 1. 

```python
import numpy as np
from PyQCodes.chan.param import ParameterizationABC


class MyParametization(ParameterizationABC):
    def numb_variables(self, nsize):
        # Doesn't depend on size of density matrix.
        return 1
        
    def bounds(nsize):
        # Doesn't depend on size of density matrix.
        return [(0, 1.)]
    
    def rho_from_vec(vec, nsize):
        delta = vec[0]
        diagonal = [vec[0]] + [0.] * (nsize - 1) + [1 - vec[0]]
        return np.diag(diagonal) # Construct the diagonal matrix.
        
    def random_vectors(nsize, _):
        # Return a random number from zero to one.
        return np.random.uniform(0, 1)
    

# It can be used as follows.
channel.optimize_coherent(n=2, rank=4, optimizer="diffev", 
                          param=MyParametization, maxiter=100, disp=True)
```
Note that if "random_vectors" is not implemented, then lipschitz parameter 
doesn't work.
     
Lipschitz
---------
The coherent information is lipschitz with lipschitz constant 1. Using the 
algorithm found in paper [^2]. 
The lipschitz parameter will generate initial guesses based on the algorithm. 
It will generate random density matirces and will save the random density 
matrix if it satisfies the lipschitz condition to be used as an initial guess.


References
==========
[^1]: Dephrasure channel and superadditivity of coherent information. By F.Leditzky, D. Leung and G. Smith.

[^2]: Global optimization of Lipschitz functions. By C. Malherbe and N. Vayatis.
