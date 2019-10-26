from PyQCodes.channel import AnalyticQChan
from PyQCodes.density_mat import OverParameterization, CholeskyDecomposition
from PyQCodes._optimize import optimize_via_sampling
import numpy as np


r""" Test optimization routines from "QCorr._optimize.py" """



def test_optimizing_overparm_choleskly():
    r"""Test that both styles of parameterizations give the same answer."""
    p = np.random.uniform(0., 0.5)
    q = np.random.uniform(0., 0.5)

    krauss_1 = np.array([[1., 0.], [0., 1.], [0., 0.]], dtype=np.complex128)
    krauss_2 = np.array([[1., 0.], [0., -1.], [0., 0.]], dtype=np.complex128)
    krauss_4 = np.array([[0., 0.], [0., 0.], [1., 0.]], dtype=np.complex128)
    krauss_5 = np.array([[0., 0.], [0., 0.], [0., 1.]], dtype=np.complex128)
    numb_krauss_ops = 4
    krauss_ops = [krauss_1 * np.sqrt((1 - p) * (1 - q)),
                  krauss_2 * np.sqrt((1 - q) * p),
                  np.sqrt(q) * krauss_4,
                  np.sqrt(q) * krauss_5]
    krauss_ops = np.array(krauss_ops)

    channel = AnalyticQChan(1, krauss_ops)

    for n in [1, 2]:
        for r in range(2, 2**n + 1):
            def minimize_coh(vec, rank=None, overparam=False):
                if overparam:
                    rho = OverParameterization.rho_from_vec(vec, n)
                else:
                    rho = CholeskyDecomposition.rho_from_vec(vec, n, rank)
                result = channel.coherent_information(rho, n)
                # print(result / n)
                return result

            results_o = optimize_via_sampling(minimize_coh, 10, 2**n,
                                              OverParameterization.bounds(n, r),
                                              use_slsqp=True)
            results_c = optimize_via_sampling(minimize_coh, 10, 2**n,
                                              CholeskyDecomposition.bounds(n, r),
                                              use_slsqp=True)
            assert results_c.max_value == results_o.max_value


if __name__ == "__main__":
    test_optimizing_overparm_choleskly()
