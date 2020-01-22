r"""
The MIT License

Copyright (c) 2019-Present PyQCodes - Software for investigating
coherent information and optimization-based quantum error-correcting codes.
PyQCodes is jointly owned equally by the University of Guelph (and its employees)
and Huawei, funded through the Huawei Innovation Research.

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
import numpy as np

r""" 
Contains optimization routines for optimizing over density matrices 
and over choi-matrices, as semi-definite problems.
"""


__all__ = ["positive_coherent_info_callback"]


def positive_coherent_info_callback(x_k, func_val, convergence):
    print("func val ", func_val)
    if func_val < -1e-8:
        return True
    return False


def _lipschitz_optimizing_condition():
    pass


def initial_guesses_lipschitz(f, nsize, rank, param, chan, n, numb_samples=10, numb_tries=100):
    r"""


    Parameters
    ----------
    f : callable
        Objective, Lipschitz Function that takes input from argument 'rho_sampler'.

    param : callable
        Function that returns uniformly random density states.

    numb_samples : int
        Number of samples returned.

    numb_tries : int
        Number of random samples to try.

    Returns
    -------


    References
    ----------
    Based on the 2017 paper, "Global Optimization of Lipschitz Functions"
    """
    # Obtain single random vectors turn into density matrix and evaluate on objective function.
    samples_vec = [param.random_vectors(nsize, rank)]
    print(nsize, rank)
    samples_rho = [param.rho_from_vec(samples_vec[0], nsize, rank)]
    func_evals = [-f(samples_vec[0], nsize, rank, param, chan, n)]
    counter = 0
    success = False
    f_max = func_evals[0]
    while (counter < numb_tries) and len(samples_vec) < numb_samples:
        vec = param.random_vectors(nsize, rank)
        rho = param.rho_from_vec(vec, nsize, rank)

        bounds = [func_evals[i] + np.sum(np.abs(rho - samples_rho[i]))
                  for i in range(0, len(samples_vec))]
        if f_max <= np.min(bounds):
            samples_vec.append(vec)
            samples_rho.append(rho)
            func_evals.append(-f(vec, nsize, rank, param, chan, n))
            f_max = np.max(func_evals)
        counter += 1

    if len(samples_vec) == numb_samples:
        success = True
    return samples_vec, func_evals, success
