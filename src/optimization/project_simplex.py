"""
    Projection of a point on to the probability simplex

    Author: Akshay Seshadri
"""

import numpy as np
import scipy as sp
from scipy import optimize

def project_on_simplex_bisection(v):
    """
        Projects the point v \in R^n on to the probability simplex C = {x \in R^n | x >= 0, \sum_i x_i = 1}

        An analytical expression for the same is not available, but simple numerical procedures are. We follow the method provided in
        Wang & Carreira-Perpinan (2013), which is non-iterative.

        Verified the projection by comparing with functions given on https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    """
    u = np.sort(v)[::-1]        # sort the components of v in descending order
    u_cumsum = np.cumsum(u)

    # rho = max{1 <= j <= n | u_j + (1 - \sum_{i = 1}^j u_i)/j > 0}
    # we find this by bisection
    j1, j2 = 0, u.size - 1
    if u[j2] + (1. - u_cumsum[j2])/(j2 + 1.) > 0:     # if the right-most index satisfies the condition, then clearly it is rho
        rho = j2
    else:
        while j2 > j1 + 1:                            # when j1 and j2 differ by 0 or 1, rho will be equal to j1
            jm = (j1 + j2)//2
            if u[jm] + (1. - u_cumsum[jm])/(jm + 1.) < 0:
                j2 = jm
            else:
                j1 = jm
        rho = j1

    lambda_rho  = (1. - u_cumsum[rho]) / (rho + 1.)   # since Python is zero-indexed, we need to divide by rho + 1

    return np.maximum(v + lambda_rho, np.zeros_like(v))
