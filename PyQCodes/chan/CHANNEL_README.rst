==================
Quantum Channels
==================
.. _channel_readme:

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

Minimum Fidelity
----------------



Algorithm Information
=====================

Parameterization
----------------

Lipschitz Properties
--------------------
