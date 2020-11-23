"""
    Projects the given Hermitian matrix on the set of density matrices (positive, trace one operators).

    Author: Akshay Seshadri
"""

import numpy as np

import project_root # noqa
from src.optimization.project_simplex import project_on_simplex_bisection

def project_on_density_matrices_flattened(V):
    """
        Projects a Hermitian matrix V on the set of density matrices X = {A \in C^{N x N} | A^\dagger = A, A >=0, tr(A) = 1}.

        Note that a Hermitian matrix must be provided. Further, the input can be the matrix or its flattened vector (row-major/C-style flattening
        assumed, which is the default of numpy). Checks are not done to ensure this is the case.

        The output is a (row-major) flattened vector of the projection.
    """
    n = int(np.sqrt(V.size))                                        # dimension of V

    V = V.reshape((n, n))                                           # V as a matrix

    V_eigvals, V_eigvecs = np.linalg.eigh(V)                        # get the spectral decomposition of V

    # project on to the density matrices by projecting the eigenvalues of V on the probability simplex
    Pi_V_eigvals = project_on_simplex_bisection(V_eigvals)

    # construct the projection form the eigevectors of V and the projected eigenvalues: Pi_V = U D_Pi U^\dagger
    Pi_V = V_eigvecs.dot( np.diag(Pi_V_eigvals).dot( V_eigvecs.T.conj() ) )

    # return the (row-major) flattened output
    return Pi_V.ravel()
