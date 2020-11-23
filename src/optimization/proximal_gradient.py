"""
    Implements the proximal gradient method.

    Author: Akshay Seshadri
"""

import numpy as np

###------------ Line search: Implement when the Lipschitz constant is not known, or if you expect "local improvements" even when it is known
def minimize_proximal_gradient_nesterov(f, P, gradf, prox_lP, xo, tol = 1e-6, return_error = False):
    """
        Minimize the objective function f + g using an accelerated proximal gradient method given by Nesterov.

        Reference: https://www.mit.edu/~dimitrib/PTseng/papers/apgm.pdf
                   https://www.csie.ntu.edu.tw/~b97058/tseng/papers/apeb.pdf

        Arguments:
            - f            : a differentiable proper convex function
            - P            : a proper convex function
            - gradf        : gradient of f
            - prox_lP      : proximal operator of P with parameter 'l'
            - xo           : initial condition used for optimization
            - tol          : The tolerance is used in the optimization
            - return_error : returns the error if True, along with the optimum

        Some definitions (we specialize to the case of proximal gradient by choosing D to be the Euclidean norm):
            l_F(x, y) = f(y) + <gradf(y), x - y> + P(x)
            D(x, y)   = 0.5 ||x - y||_2^2
    """
    # ensure that the while loop doesn't run ad infinitum
    count = 0
    max_count = 1e5

    # initialize
    x_k = xo
    z_k = xo
    theta_k = 1.

    # line search for "estimating" the Lipschitz constant L
    L = 1.                                                     # some initial guess
    beta = 1.5                                                 # we increase L until appropriate condition is satisfied

    # perform the first iteration so that we can enter the while loop
    y_k      = (1. - theta_k) * x_k + theta_k * z_k
    z_k_next = prox_lP(z_k - gradf(y_k) / (theta_k * L), 1./(theta_k * L), tol)
    x_k_next = (1. - theta_k) * x_k + theta_k * z_k_next

    # ensure that the "inner loop" doesn't run ad infinitum
    count_inner = 0
    max_count_inner = 20

    l_F = lambda x, y: f(y) + np.dot(gradf(y), x - y) + P(x)

    while f(x_k_next) + P(x_k_next) > ( (1. - theta_k)*l_F(x_k, y_k) + theta_k * l_F(z_k_next, y_k)\
                                            + 0.5*theta_k**2 * L * np.linalg.norm(z_k_next - z_k)**2 ) and count_inner < max_count_inner:
        L *= beta

        y_k      = (1. - theta_k) * x_k + theta_k * z_k
        z_k_next = prox_lP(z_k - gradf(y_k) / (theta_k * L), 1./(theta_k * L), tol)
        x_k_next = (1. - theta_k) * x_k + theta_k * z_k_next

        count_inner += 1

    if count_inner == 0:
        L /= beta

    theta_k_next = 0.5*np.sqrt(theta_k**4 + 4.*theta_k**2) - 0.5*theta_k**2

    # start the minimization
    while np.linalg.norm(x_k_next - x_k) > tol and count < max_count:
        # update the variables
        x_k     = x_k_next
        z_k     = z_k_next
        theta_k = theta_k_next

        # perform the line search to "estimate" the Lipschitz constant, and update the iterates
        y_k      = (1. - theta_k) * x_k + theta_k * z_k
        z_k_next = prox_lP(z_k - gradf(y_k) / (theta_k * L), 1./(theta_k * L), tol)
        x_k_next = (1. - theta_k) * x_k + theta_k * z_k_next

        count_inner = 0

        while f(x_k_next) + P(x_k_next) > ( (1. - theta_k)*l_F(x_k, y_k) + theta_k * l_F(z_k_next, y_k)\
                                            + 0.5*theta_k**2 * L * np.linalg.norm(z_k_next - z_k)**2 ) and count_inner < max_count_inner:
            L *= beta

            z_k_next = prox_lP(z_k - gradf(y_k) / (theta_k * L), 1./(theta_k * L), tol)
            x_k_next = (1. - theta_k) * x_k + theta_k * z_k_next

            count_inner += 1

        if count_inner == 0:
            L /= beta

        theta_k_next = 0.5*np.sqrt(theta_k**4 + 4.*theta_k**2) - 0.5*theta_k**2

        count += 1

    if not return_error:
        return x_k_next
    else:
        return (x_k_next, np.linalg.norm(x_k_next - x_k))
