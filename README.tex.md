PyQCodes
========
<a href='https://docs.python.org/3.6/'><img src='https://img.shields.io/badge/python-3.6-blue.svg'></a>

PyQCodes provides a set of functions to investigate coherent information,
minimum fidelity, certain optimization-based codes and stabilizer codes.

Guide To Do
-----------
* Quantum Channels. See [file for more info](PyQCodes/chan/CHANNEL_README.md).
    - Compute Channel, Adjoint of Channel.
    - Compute Complementary Channel and Adjoint of Complementary Channel.
    - Optimize Coherent Information and Minimum Fidelity.
    - Serial/Parallel concatenate two channels .
* Stabilizer Codes.  See [file for more info](PyQCodes/CODE_README.md).
    - Find Logical Operators.
    - Apply Encoding, Measurement, and Decoding Circuit.
* Optimization-Based Codes.  See :ref:`_code_readme`.
    - Optimizes the average fidelity over Recover/Encoding operators.
    - Effective Channel Method of Stabilizer Codes.

Examples
--------

Getting Started
===============

Prerequisites
-------------

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

