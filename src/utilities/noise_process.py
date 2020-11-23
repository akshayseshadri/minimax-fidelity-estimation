"""
    Implements various forms of quantum noise.

    Author: Akshay Seshadri
"""

import numpy as np

def depolarizing_channel(rho, p = 0.1):
    """
        Completely mixes state (density matrix) with a certain probability 'p'.

        N(rho) = (1 - p) rho + p I / d

        where 'd' is the dimension of the system.

        Reference: Section 4.7.4, Wilde Book; https://en.wikipedia.org/wiki/Quantum_depolarizing_channel

        The density matrix is assumed to be flattened.
    """
    d = int(np.sqrt(rho.size))

    # the noisy state
    N_rho = (1 - p) * rho + p * np.eye(d).ravel() / float(d)

    return N_rho
