from PyQCodes.utils import decompose_matrix
from PyQCodes.param import OverParam, CholeskyParam, ParameterizationABC
from PyQCodes._optimize import initial_guesses_lipschitz

import numpy as np
from scipy.linalg import logm

from scipy.optimize import minimize, differential_evolution
from pathos.multiprocessing import ProcessingPool as Pool

r""" 
Computing and optimizing the coherent information based on AnalyticQChannel. 
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
    return channel.fidelity_two_states(rho)


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


def _optimize_using_bfgs(x0, bounds, args, maxiter, eps, ftol, fobjective):
    options = {"disp": False, "eps": eps, "ftol": ftol, "maxiter":maxiter}
    return minimize(fobjective, x0, method="L-BFGS-B", bounds=bounds, options=options, args=args)


def _use_multi_processes(samples, bounds, args, maxiter, eps, ftol, fobjective, pool_number):
    r"""Optimize over the samples using multi-processing."""
    obj_func = lambda x: _optimize_using_bfgs(x, bounds, args, maxiter, eps, ftol, fobjective)
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


def _optimize_samples_bfgs(samples, bounds, args, maxiter, eps, ftol, fobjective):
    r"""
    Optimize over the samples using BFGS.

    Parameters
    ----------
    samples
    bounds
    args

    Returns
    -------
    (float, bool, array) :

    """
    optimal_val = -1e10
    for s in samples:
        optimize = _optimize_using_bfgs(s, bounds, args, maxiter, eps, ftol, fobjective)
        if optimize["fun"] > optimal_val:
            optimal_val = optimize
            optimal_x_val = optimize["x"]
            success = optimize["success"]
    return optimal_val, success, optimal_x_val


def _optimize_differential_evol(samples, bounds, args, popsize, maxiter, fobjective):
    r""" Optimize using differential evolution."""
    optimize = differential_evolution(fobjective, bounds=bounds, mutation=1.5,
                                      recombination=0.75, strategy="rand2exp", popsize=popsize,
                                      maxiter=maxiter, disp=True, workers=2, init=samples,
                                      args=args)
    optimal_val = optimize["fun"]
    optimal_x_val = optimize["x"]
    success = optimize["success"]
    return optimal_val, success, optimal_x_val


def optimize_procedure(channel, n, rank, optimizer="diffev", param="overparam",
                       objective="coherent", lipschitz=0,
                       use_pool=0, maxiter=50, samples=()):
    r"""

    Parameters
    ----------
    channel :
    n : int

    rank : int

    optimizer : str

    param : ParameterizationABC

    lipschitz : int

    use_pool : int

    maxiter : int

    samples : list

    Returns
    -------

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
                                                                          objective_func)

    elif optimizer == "slsqp":
        assert lipschitz != 0, "SLSQP parameter requires 'lipschitz' parameter to be nonzero."
        # Epsilon and ftol parameters for optimizer.
        eps, ftol = 1e-5, 1e-8

        # Optimize over samples provided by lipschitz either using multi-processing or for-loop.
        if use_pool != 0:
           optimal_val, success, optimal_x_val = _use_multi_processes(samples, bounds, args,
                                                                      maxiter, eps, ftol,
                                                                      objective_func, use_pool)
        else:
           optimal_val, success, optimal_x_val = _optimize_samples_bfgs(samples, bounds, args,
                                                                        maxiter, eps, ftol,
                                                                        objective_func)
    else:
        raise TypeError("Optimizer should be 'diffev' or 'slsqp'.")

    if objective == "coherent":
        optimal_val *= -1.  # Multiply by negative one because we are minimizing but want max.

    # Update the output
    output = {"optimal_rho": None, "optimal_val": None, "method": optimizer,
              "lipschitz": use_lipschitz, "success" : success, "objective":objective}
    output["optimal_rho"] = param.rho_from_vec(optimal_x_val, nsize, rank)
    output["optimal_val"] = optimal_val
    return output
