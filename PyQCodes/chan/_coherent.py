r"""
The MIT License

Copyright (c) 2019-Present PyQCodes

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
from PyQCodes.chan.param import OverParam, CholeskyParam, ParameterizationABC
from PyQCodes._optimize import initial_guesses_lipschitz

import numpy as np
from scipy.optimize import minimize, differential_evolution
from pathos.multiprocessing import ProcessingPool as Pool

r""" 
Contains functions to optimize the coherent information or minimum fidelity.
"""

__all__ = ["optimize_procedure"]


def _objective_func_coherent(vec, nsize, rank, param, channel, n):
    # Maximize coherent information of channel.
    rho = param.rho_from_vec(vec, nsize, rank)
    return - channel.coherent_information(rho, n)


def _objective_func_fidelity(vec, nsize, _, param, channel, n):
    # Rank here is redundant.
    # Minimize channel fidelity.
    rho = param.rho_from_vec(vec, nsize, rank=1)
    return channel.fidelity_two_states(rho, n)


def _choose_objective_func(objective):
    if objective == "coherent":
        return _objective_func_coherent
    elif objective == "fidelity":
        return _objective_func_fidelity
    else:
        raise TypeError("Objective has to be wither 'coherent' or 'fidelity'.")


def _get_parameterization_scheme_density(param):
    r"""Get parameterization static class from the param string"""
    if not (isinstance(param, str) or isinstance(param, ParameterizationABC)):
        raise TypeError("Param should either be string or class ParameterizationABC.")
    if isinstance(param, ParameterizationABC):
        return param
    if param == "overparam":
        paramet = OverParam
    elif param == "cholesky":
        paramet = CholeskyParam
    else:
        raise TypeError("Parameterization should be 'overparam' or 'cholesky'.")
    return paramet


def _optimize_using_slsqp(x0, bounds, args, maxiter, eps, ftol, fobjective, disp):
    r"""Optimize using SLSQP."""
    options = {"disp": disp, "eps": eps, "ftol": ftol, "maxiter":maxiter}
    return minimize(fobjective, x0, method="slsqp", bounds=bounds, options=options, args=args)


def _use_multi_processes(samples, bounds, args, maxiter, eps, ftol, fobjective, pool_number, disp):
    r"""Optimize over the samples using multi-processing."""
    obj_func = lambda x: _optimize_using_slsqp(x, bounds, args, maxiter, eps, ftol, fobjective, disp)
    pool = Pool(processes=pool_number)
    results = pool.map(obj_func, samples)

    # Initialize parameters for finding maximas over the results.
    maxima = -1e10
    optimal_x_val = None
    success = False
    for r in results:
        if -r["fun"] > maxima:
            maxima = r["fun"]
            optimal_x_val = r["x"]
            success = r["success"]
    return maxima, success, optimal_x_val


def _optimize_samples_slsqp(samples, bounds, args, maxiter, eps, ftol, fobjective, disp):
    r"""
    Optimize over the samples using SLSQP.

    Parameters
    ----------
    samples : list
        List of samples of the parameterization of density matrix.
    bounds : list
        Bounds on the variables of the parameterization of density matrices.
    args : tuple
        The tuple (nsize, rank, param, channel, n) where nsize is the number of variables to
        parameterize density matrix, the rank of a density matrix, param is the string of the
        parameterization, channel is the AnalyticQChan, and n is how many times the channel was
        tensored.
    maxiter : int
        The maximum number of iterations.
    eps : float
        For SLSQP, step size used for approximation of the Jacobian.
    ftol : float
        For SLSQP, precision goal for the objective function in the stopping criterion.
    fobjective : callable
        The objective function to be optimized. Either fidelity or coherent information.
    disp : bool
        Display the process of the optimization procedure.

    Returns
    -------
    (float, bool, np.ndarray) :
        Returns the optimal solution found, whether it converges or not and the variables of the
        parameterization corresponding to the optimal solution.
    """
    optimal_val = -1e10
    for s in samples:
        optimize = _optimize_using_slsqp(s, bounds, args, maxiter, eps, ftol, fobjective, disp)
        if optimize["fun"] > optimal_val:
            optimal_val = optimize
            optimal_x_val = optimize["x"]
            success = optimize["success"]
    return optimal_val, success, optimal_x_val


def _optimize_differential_evol(samples, bounds, args, popsize, maxiter, fobjective, disp):
    r"""
    Optimize using differential evolution.

    Parameters
    ----------
    samples : list
        List of samples of the parameterization of density matrix.

    bounds : list
        Bounds on the variables of the parameterization of density matrices.
    args : tuple
        The tuple (nsize, rank, param, channel, n) where nsize is the number of variables to
        parameterize density matrix, the rank of a density matrix, param is the string of the
        parameterization, channel is the AnalyticQChan, and n is how many times the channel was
        tensored.
    popsize : int
        The number of initial guesses.
    maxiter : int
        The maximum number of iterations.
    fobjective : callable
        The objective function to be optimized. Either fidelity or coherent information.
    disp : bool
        Display the process of the optimization procedure.

    Returns
    -------
    (float, bool, np.ndarray) :
        Returns the optimal solution found, whether it converges or not and the variables of the
        parameterization corresponding to the optimal solution.
    """
    optimize = differential_evolution(fobjective, bounds=bounds, mutation=1.5,
                                      recombination=0.75, strategy="rand2exp", popsize=popsize,
                                      maxiter=maxiter, disp=disp, workers=2, init=samples,
                                      args=args)
    optimal_val = optimize["fun"]
    optimal_x_val = optimize["x"]
    success = optimize["success"]
    return optimal_val, success, optimal_x_val


def optimize_procedure(channel, n, rank, optimizer="diffev", param="overparam",
                       objective="coherent", lipschitz=0, use_pool=0, maxiter=50, samples=(),
                       disp=False):
    r"""
    Optimization Procedure to Optimize Over Fixed Rank density matrices of Coherent or Fidelity.

    This optimization procedure does not necessarily find the global optima.

    Parameters
    ----------
    channel : AnalyticQChan
        The quantum channel object.
    n : int
        The number of qubits.
    rank : int
        The rank of the density matrices.
    optimizer : str
        If "diffev": Then optimization is done using the differential evolution function from scipy.
        If "slsqp": Then optimization is done using the SLSQP function from scipy.
    param : str or ParameterizationABC
        The parameterization of the fixed rank density matrices.
        If "overparam": Then parameterization using the overparemterization method.
        if "choleskly" Then parameterization is done using the choleskly parameterization.
        If object of type ParameterizationABC: Then paramterization is done using user specified.
    lipschitz : int, default zero
        If non-zero integer, it will use the lipschitz properties to find proper initial guesses.
    use_pool : int, default zero
        If positive, then it uses the multiprocessing python package to speed up calculations.
    maxiter : int
        The maximum number of iterations used in the optimizer.
    samples : list
        If it is not empty, then it adds these samples to be used for initial guesses.
    disp : bool
        True, then print during the optimization procedure.

    Returns
    -------
    dict :
        The result is a dictionary with fields:

            optimal_rho : np.ndarray
                The density matrix of the optimal solution.
            optimal_val : float
                The optimal value of either coherent information or fidelity.
            method : str
                Either diffev or slsqp.
            success : bool
                True if optimizer converges.
            objective : str
                Either coherent or fidelity
            lipschitz : bool
                True if uses lipschitz properties to find initial guesses.

    """
    param = _get_parameterization_scheme_density(param)
    objective_func = _choose_objective_func(objective)
    assert isinstance(use_pool, int), "Number of pool processing should be integer."
    assert use_pool >= 0., "Number of pool processing units should be positive or zero."
    if len(samples) != 0:
        assert isinstance(samples, list)
    assert lipschitz >= 0.
    assert isinstance(lipschitz, int)
    if len(samples) == 0 and optimizer == "slsqp":
        assert lipschitz > 0, "For SLSQP and empty samples list, then lipschitz must be greater " \
                              "than zero."

    nsize = channel.input_dimension  # Get the numb rows/cols of density matrix (input of channel)
    bounds = param.bounds(nsize, rank)  # Get the bounds for the optimization variables.
    args = (nsize, rank, param, channel, n)  # Get arguments for objective-function/coherent info.
    use_lipschitz = False

    if lipschitz != 0:
        lip_samps, func_evals, success = initial_guesses_lipschitz(objective_func, *args,
                                                                   numb_samples=lipschitz,
                                                                   numb_tries=1000)
        # Add lipschitz samples to samples.
        samples = list(samples) + lip_samps
        use_lipschitz = True
        assert success, "Lipschitz sampler could not find 'lipschitz' number of initial guesses. " \
                        "Try smaller number."

    if optimizer == "diffev":
        # Use differential_evolution sampler 'latinhypercube' or use the lipschitz for initial guess
        popsize = 50
        init = "latinhypercube"
        # If User provided own samples or using lipschitz sampler.
        if len(samples) != 0 or lipschitz != 0:
            init = np.array(samples)
            popsize = len(samples)
        optimal_val, success, optimal_x_val = _optimize_differential_evol(init, bounds, args,
                                                                          popsize, maxiter,
                                                                          objective_func, disp)

    elif optimizer == "slsqp":
        assert lipschitz != 0, "SLSQP parameter requires 'lipschitz' parameter to be nonzero."
        # Epsilon and ftol parameters for optimizer.
        eps, ftol = 1e-5, 1e-8

        # Optimize over samples provided by lipschitz either using multi-processing or for-loop.
        if use_pool != 0:
           optimal_val, success, optimal_x_val = _use_multi_processes(samples, bounds, args,
                                                                      maxiter, eps, ftol,
                                                                      objective_func, use_pool,
                                                                      disp)
        else:
           optimal_val, success, optimal_x_val = _optimize_samples_slsqp(samples, bounds, args,
                                                                         maxiter, eps, ftol,
                                                                         objective_func, disp)
    else:
        raise TypeError("Optimizer should be 'diffev' or 'slsqp'.")

    if objective == "coherent":
        optimal_val *= -1.  # Multiply by negative one because we are minimizing but want max.

    # Update the output
    output = {"optimal_rho": None, "optimal_val": None, "method": optimizer, "lipschitz":
              use_lipschitz, "success" : success, "objective":objective}
    output["optimal_rho"] = param.rho_from_vec(optimal_x_val, nsize, rank)
    output["optimal_val"] = optimal_val
    return output
