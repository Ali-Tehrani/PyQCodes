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
Consider the dephrasure channel <img src="/PyQCodes/chan/tex/062ac33007e50544849e2c07322e45b6.svg?invert_in_darkmode&sanitize=true" align=middle width=145.93951679999998pt height=24.65753399999998pt/> that maps a qubit to a qutrit. It was
 introduced in the paper [^1]. It has the following action on qubit density 
 matrix <img src="/PyQCodes/chan/tex/6dec54c48a0438a5fcde6053bdb9d712.svg?invert_in_darkmode&sanitize=true" align=middle width=8.49888434999999pt height=14.15524440000002pt/>:
 
 <p align="center"><img src="/PyQCodes/chan/tex/b1ddc8fa17ee69ad15f4eeb9bbaf8501.svg?invert_in_darkmode&sanitize=true" align=middle width=363.52370175pt height=16.438356pt/></p>
 
 It has four kraus operators, the first two relating to I and Z and the last 
 two relating to erasure <img src="/PyQCodes/chan/tex/3089829b0ce46137aaf0c567f7ef5ebe.svg?invert_in_darkmode&sanitize=true" align=middle width=37.22615654999999pt height=24.65753399999998pt/>.
 
 ```python
import numpy as np
from PyQCodes.chan.channel import AnalyticQChan
 
# Set up kraus operators
p, q = 0.1, 0.2
krauss_1 = np.array([[1., 0.], [0., 1.], [0., 0.]])
krauss_2 = np.array([[1., 0.], [0., -1.], [0., 0.]])
krauss_4 = np.array([[0., 0.], [0., 0.], [1., 0.]])
krauss_5 = np.array([[0., 0.], [0., 0.], [0., 1.]])
krauss_ops = [krauss_1 * np.sqrt((1 - p) * (1 - q)),
              krauss_2 * np.sqrt((1 - q) * p),
              np.sqrt(q) * krauss_4,
              np.sqrt(q) * krauss_5]

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
The AnalyticQChan object can compute the channel <img src="/PyQCodes/chan/tex/4632c974a75d4fb59e15d2ab904ff1f6.svg?invert_in_darkmode&sanitize=true" align=middle width=37.19417294999999pt height=24.65753399999998pt/>, the 
complementary channel <img src="/PyQCodes/chan/tex/5f3ea9a3c844a76fb683726e7ec94633.svg?invert_in_darkmode&sanitize=true" align=middle width=43.890740849999986pt height=24.65753399999998pt/>, and adjoint of each of these 
matrices. And it is able to compute channel <img src="/PyQCodes/chan/tex/3efd32760af1e5bae1c960ea5bf31b05.svg?invert_in_darkmode&sanitize=true" align=middle width=34.309900349999985pt height=26.17730939999998pt/> tensored n times (although it grows fast!).

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

Two channels <img src="/PyQCodes/chan/tex/aa23b476f1bef42877182d581b7dd541.svg?invert_in_darkmode&sanitize=true" align=middle width=20.040066749999987pt height=22.465723500000017pt/> and <img src="/PyQCodes/chan/tex/898247703c06e39cfc2265a15849349c.svg?invert_in_darkmode&sanitize=true" align=middle width=20.040066749999987pt height=22.465723500000017pt/> can be used to create a third channel <img src="/PyQCodes/chan/tex/ac26e85ef6b8399a2dec41b31b889a57.svg?invert_in_darkmode&sanitize=true" align=middle width=56.42701514999999pt height=22.465723500000017pt/> via serial concatenation.
```python
chan1 = AnalyticQChan(kraus1, [1, 1], 2, 3)
chan2 = AnalyticQChan(kraus2, [1, 1], 2, 3)
new_channel = chan2 + chan1
```

Similarly, two channel can create a third tensored channel <img src="/PyQCodes/chan/tex/239bdd26bae048393d9077bbe99969cc.svg?invert_in_darkmode&sanitize=true" align=middle width=60.99323834999999pt height=22.465723500000017pt/> via parallel concatenation.
```python
new_channel = chan2 * chan1
# Note that the new channel will always assume it will map one qubit to one qubit.
```

Note that The complementary channel <img src="/PyQCodes/chan/tex/58f51d09f8378a6f3489037e62d8c81a.svg?invert_in_darkmode&sanitize=true" align=middle width=21.78450779999999pt height=22.465723500000017pt/> maps a density matrix to the environment system, 
here it is defined as the matrix with ith jth entries as <img src="/PyQCodes/chan/tex/df282f8934e95b9d00b6ab0684924d90.svg?invert_in_darkmode&sanitize=true" align=middle width=82.66960515pt height=31.780732499999996pt/>  

Coherent Information
--------------------
The coherent information of a channel <img src="/PyQCodes/chan/tex/2df4fb76c1e8861fde00e5c2553de81d.svg?invert_in_darkmode&sanitize=true" align=middle width=42.617907749999986pt height=24.65753399999998pt/> is defined to be

<p align="center"><img src="/PyQCodes/chan/tex/36f64438508a7129a7db6deda1aa5b30.svg?invert_in_darkmode&sanitize=true" align=middle width=251.23687214999998pt height=24.4292268pt/></p>

Maximization is done by parameterizing rank-k density matrices (See 
[below](#parameterization)). Using the dephrasure example above.

```python
results = channel.optimize_coherent(n=2, rank=4, optimizer="diffev", 
                                    lipschitz=50, use_pool=3, maxiter=100, 
                                    disp=True)
print("Is successful: ", results["success"])
print("Optimal rho", results["optimal_rho"])
print("Optimal Value", results["optimal_val"])
```
This will optimize coherent information of <img src="/PyQCodes/chan/tex/6e8fd6045ec7bd4ef7fb13d25d31f890.svg?invert_in_darkmode&sanitize=true" align=middle width=32.73642404999999pt height=26.76175259999998pt/> over rank four (the highest rank) density matrices. It is generally recommended to
optimize over the highest rank. The optimization procedure will use "differential_evolution" from Scipy. It will generate 50 initial population 
sizes using the lipschitz sampler. It will execute 3 processes from the 
multiprocessing library to run it faster and will have a maximum iteration of 100.
The disp keyword will print the results as the optimization procedure 
progresses.

See [documentation](channel.py) of the method 'optimize_coherent' 
for more info.

It is highly recommended to use lipschitz sampler and use_pool and to optimize over the highest rank.
Furthermore, using lipschitz sampler and optimizer="slsqp" generally outperforms "differential_evolution".
For large n, you may need to reconstruct the AnalyticQChan using the sparse 
keyword as True. This does not guarantee to find the global optima, but it 
will find a local maxima.


Minimum Fidelity
----------------
The fidelity of a quantum state <img src="/PyQCodes/chan/tex/6dec54c48a0438a5fcde6053bdb9d712.svg?invert_in_darkmode&sanitize=true" align=middle width=8.49888434999999pt height=14.15524440000002pt/> with respect to a channel 
<img src="/PyQCodes/chan/tex/c15fcfc9ce9b5c7b55995db1cfb727f9.svg?invert_in_darkmode&sanitize=true" align=middle width=15.90987419999999pt height=22.465723500000017pt/> is:

<p align="center"><img src="/PyQCodes/chan/tex/7cd5a45196c6c2f886cf838ad6e966b8.svg?invert_in_darkmode&sanitize=true" align=middle width=242.54683695pt height=39.452455349999994pt/></p>
The minimum fidelity is the minimum of the fidelity of a channel over rank one density matrices.

Here, the minimum fidelity over the dephrasure channel <img src="/PyQCodes/chan/tex/6e8fd6045ec7bd4ef7fb13d25d31f890.svg?invert_in_darkmode&sanitize=true" align=middle width=32.73642404999999pt height=26.76175259999998pt/> is applied here.

```python
results = channel.optimize_fidelity(n=2, optimizer="diffev", lipschitz=50, 
                                    use_pool=3, maxiter=100, disp=True)
print("Is successful: ", results["success"])
print("Optimal rho", results["optimal_rho"])
print("Optimal Value", results["optimal_val"])
```

The same parameters as optimizing coherent information is applied here.

See [documenation](channel.py) of the method 'optimize_fidelity' for more info.


Algorithm Information
=====================

Parameterization
----------------
There are two different parameterization done, OverParameterization and 
Cholesky Parameterization.


Lipschitz Properties
--------------------


References
==========
[^1] : Dephrasure channel and superadditivity of coherent information. By F.Leditzky, D. Leung and G. Smith.