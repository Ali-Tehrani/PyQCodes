.. _channel_readme:
==================
Quantum Channels
==================

Overview
========

This module contains the AnalyticQChan object which models the quantum
channels as a kraus operator (can also do Choi matrices but is not recommended).

It has the following properties and methods that can be of used.

- Compute the channel, complementary channel (called entropy exchange here),
the adjoint of a channel and adjoint of complementary channel.
- Can add (serial concatenate), multiply (parallel concatenate) different
channels together.
- Compute the Von Neumann entropy and fidelity of a state wrt to a channel.
- Maximizes the coherent Information of a channel.
- Minimizes the minimum fidelity of a channel.
- Model kraus operators as sparse matrices.
- Calculate the average fidelity of a channel with respect to an ensemble.


Examples
========

General Properties
------------------

Coherent Information
--------------------
This example shows how to calculate the one-shot (n=1) coherent information of the dephasing channel. It will be optimized using Scipy's Differential Evolution solver ("diffev") over rank two density matrices with a maximum iteration of 200.

.. codeblock:: python

   p = 0.25  
   kraus_ops = [k0, k1, k2, k3]
   numb_qubits = [1, 1]  # Accepts one qubits and outputs one qubits.
   dim_in = 2  # Dimension of single qubit hilbert space is 2.
   dim_out = 2  # Dimension of single qubit hilbert space after the dephasing channel is 2.
   channel = AnalyticQChan(kraus_ops, numb_qubits, dim_in, dim_out)
   coherent = channel.optimize_coherent(n=1, rank=2, optimizer="diffev", param="overparam",
                                        maxiter=200)

The next example shows how to optimize the two-shot (n=2) coherent information of the dephrasure channel. It will be optimized using SLSQP algorithm over rank four density matrices.

.. codeblock:: python
   :linenos:

   p = 0.25  
   kraus_ops = [k0, k1, k2, k3]
   numb_qubits = [1, 1]  # Accepts one qubits and outputs one qubits.
   dim_in = 2  # Dimension of single qubit hilbert space is 2.
   dim_out = 3  # Dimension of single qubit hilbert space after the dephasing channel is 2.
   channel = AnalyticQChan(kraus_ops, numb_qubits, dim_in, dim_out, sparse=True)
   coherent = channel.optimize_coherent(n=2, rank=4, optimizer="slsqp", param="overparam", maxiter=200)

Minimum Fidelity
----------------



Algorithm Information
=====================

Parameterization
----------------

Lipschitz Properties
--------------------
