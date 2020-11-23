"""
    Given a Hermitian operator A (observable), and an unknown state sigma, finds the expectation value <A>_sigma = Tr(A sigma).

    Tomographic measurements of sigma using different N POVMs {E^{i}_k}_{k = 1}^{N_i} (1 <= i <= N) is used to estimate the expectation value.

    This is done by following Juditsky & Nemirovski's approach.
    First, we find the saddle point of the concave-convex function

    \Phi_r(sigma_1, sigma_2; phi, alpha) = Tr(A sigma_1) - Tr(A sigma_2)
                                                + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(-phi^{i}_k/alpha) (p^{i}_1)_k)
                                                    + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(phi^{i}_k/alpha) (p^{i}_2)_k)
                                                        + 2 alpha r

    where
    (p^{i}_1)_k = (Tr(E^(i)_k sigma_1) + \epsilon_o/Nm) / (1 + \epsilon_o) and
    (p^{i}_2)_k = (Tr(E^{i}_k sigma_2) + \epsilon_o/Nm) / (1 + \epsilon_o)
    are the probability distributions corresponding to the ith POVM {E^{i}_k}_{k = 1}^{N_i} with N_i elements.
    R_i > 0 is a parameter that denotes the number of observations of the ith type of measurement (i.e., ith POVM).
    There are a total of N POVMs.

    X is the set of density matrices, A is the operator whose expectation value needs to be estimated. r > 0 is a parameter.

    Then, given the saddle point sigma_1*, sigma_2*, phi*, alpha*, we can construct an estimator
    \hat{F}(\omega^{1}_1, ..., \omega^{1}_{R_1}, ... \omega^{N}_1, ..., \omega^{N}_{R_N})
                        = \sum_{i = 1}^N \sum_{l = 1}^{R_i} phi^{i}*(\omega^{i}_l) + c

    where the constant 'c' is given by the optimization problem

    c = 0.5 \max_{sigma_1} [Tr(A sigma_1) + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(-phi^{i}_k/alpha) (p^{i}_1)_k)]
         - 0.5 \max_{sigma_2} [-Tr(A sigma_2) + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(phi^{i}_k/alpha) (p^{i}_2)_k)]

    The saddle point value \Phi*(r) gives an upper bound for the confidence interval within which the error lies.

    Author: Akshay Seshadri
"""

import numpy as np
import scipy as sp
from scipy import optimize

import project_root # noqa
from src.optimization.project_density_matrices_set import project_on_density_matrices_flattened
from src.optimization.proximal_gradient import minimize_proximal_gradient_nesterov
from src.utilities.qi_utilities import generate_random_state, embed_hermitian_matrix_real_vector_space

class Expectation_Value_Estimation_Manager():
    """
        Solves the different optimization problems required for expectation value estimation using Juditsky & Nemirovski's approach.

        This involves finding a saddle point of the function

        \Phi_r(sigma_1, sigma_2; phi, alpha) = Tr(A sigma_1) - Tr(A sigma_2)
                                                + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(-phi^{i}_k/alpha) (p^{i}_1)_k)
                                                    + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(phi^{i}_k/alpha) (p^{i}_2)_k)
                                                        + 2 alpha r

        where
        (p^{i}_1)_k = (Tr(E^(i)_k sigma_1) + \epsilon_o/Nm) / (1 + \epsilon_o) and
        (p^{i}_2)_k = (Tr(E^{i}_k sigma_2) + \epsilon_o/Nm) / (1 + \epsilon_o)
        are the probability distributions corresponding to the ith POVM {E^{i}_k}_{k = 1}^{N_i} with N_i elements.
        R_i > 0 is a parameter that denotes the number of observations of the ith type of measurement (i.e., ith POVM).
        There are a total of N POVMs.

        X is the set of density matrices, A is the operator whose expectation value needs to be estimated. r > 0 is a parameter.

        Then, given the saddle point sigma_1*, sigma_2*, phi*, alpha*, we can construct an estimator
        \hat{F}(\omega^{1}_1, ..., \omega^{1}_{R_1}, ... \omega^{N}_1, ..., \omega^{N}_{R_N})
                            = \sum_{i = 1}^N \sum_{l = 1}^{R_i} phi^{i}*(\omega^{i}_l) + c

        where the constant 'c' is given by the optimization problem

        c = 0.5 \max_{sigma_1} [Tr(A sigma_1) + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(-phi^{i}_k/alpha) (p^{i}_1)_k)]
             - 0.5 \max_{sigma_2} [-Tr(A sigma_2) + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(phi^{i}_k/alpha) (p^{i}_2)_k)]

        The saddle point value \Phi*(r) gives an upper bound for the confidence interval within which the error lies.
    """
    def __init__(self, R_list, epsilon, A, POVM_list, epsilon_o, tol = 1e-6, random_seed = 1):
        """
            Assigns values to parameters and defines and initializes functions.

            We perform most of the optimization by isometrically embedding Hermitian matrices into a real vector space.
            So, we refer to the matrix form by appending '_full' to the variable name.
        """
        # initial condition for optimization algorithms uses random state
        if random_seed:
            np.random.seed(int(random_seed))

        # confidence level
        self.epsilon = epsilon

        # obtain 'r' from \epsilon
        self.r = np.log(2./epsilon)

        # constant to keep the probabilities in Born rule positive
        self.epsilon_o = epsilon_o

        # observable under consideration (before embedding)
        self.A_full = A

        # list of POVMs, one corresponding to each type of measurement
        self.POVM_list_full = POVM_list

        # dimension of the observable, density matrices (full operator/state)
        self.n = A.size
        # number of POVMs (also the number of types of measurement)
        self.N = len(POVM_list)
        # number of elements in each POVM
        self.N_list = [len(POVM) for POVM in POVM_list]

        # list of number of repetitions of each (type of) measurement
        # convert R to a floating point number if it already isn't one
        if type(R_list) != list:
            self.R_list = [float(R_list)]*self.N
        else:
            self.R_list = [float(R) for R in R_list]

        # embed all Hermitian matrices into a real vector space
        # size of A before embedding is n^2 (flattened, over complex vector space) and after embedding is also n^2 (but over a real vector space)
        self.A = embed_hermitian_matrix_real_vector_space(A)
        self.POVM_list = [[embed_hermitian_matrix_real_vector_space(E_i) for E_i in POVM] for POVM in POVM_list]

        # convert each (embedded) POVM into a matrix
        self.POVM_mat_list = [np.vstack(POVM) for POVM in self.POVM_list]

        # tolerance for all the computations
        self.tol = tol

        ### initial condition initializations
        # we use a random initial condition, since we don't know (or assume) anything about the true state
        sigma_init = generate_random_state(int(np.sqrt(self.n)), pure = False, density_matrix = True, flatten = True, random_seed = None)
        sigma_init = embed_hermitian_matrix_real_vector_space(sigma_init.astype('complex128'))

        # initalization for minimize_lagrangian_density_matrices (to be used specifically for find_density_matrices_saddle_point)
        self.mldm_sigma_ds_o = np.concatenate((sigma_init, sigma_init))

        # initialization for maximize_Phi_r_density_matrices_multiple_measurements (to be used specifically for find_alpha_saddle_point_expectation_value_estimation)
        self.mpdm_sigma_ds_o = np.concatenate((sigma_init, sigma_init))

    ###----- Finding x, y maximum of \Phi_r
    def minimize_lagrangian_density_matrices_saddle_point(self, l):
        """
            Performs the minimization \min_{sigma_1, sigma_2 \in X} L(sigma_1, sigma_2; \lambda) for any given \lambda >= 0 in the case of multiple
            measurements (POVMs).

            The lagrangian is given as

            L(sigma_1, sigma_2; \lambda) = -Tr(A sigma_1) + Tr(A sigma_2)
                                                - \lambda (\sum_{i = 1}^N R_i \log(\sum_{k = 1}^{N_i} \sqrt{(p_1)^{i}_k (p_2})^{i}_k) + r)/R_max

            where
            (p_1)^{i}_k = (Tr(E^{i}_k sigma_1) + \epsilon_o/N_i) / (1 + \epsilon_o),
            (p_2)^{i}_k = (Tr(E^{i}_k sigma_2) + \epsilon_o/N_i) / (1 + \epsilon_o)
            and N_i denotes the number of elements in the ith POVM. R_i, r > 0 are parameters. l >= 0 is the dual variable.
            R_max = max_i R_i

            Proximal gradient is used for this purpose.

            Is is expected that all the Hermitian matrices are embedded into a real vector space.
            If any result do contain matrices (by converting back the embedded vector into a matrix), these matrices
            are expected to be flattened (in row-major style).
            Operations, where possible, are performed as matrix/array operations.
        """
        # maximum number of repetitions of any type of measurement
        R_max = np.max(self.R_list)

        # we work with direct sum sigma_ds = (sigma_1, sigma_2) for use in pre-written algorithms
        # the objective function
        def Lagr(sigma_ds):
            sigma_1 = sigma_ds[0: self.n]
            sigma_2 = sigma_ds[self.n: 2*self.n]

            # start with the terms that don't depend on POVMs
            L_val = -self.A.dot(sigma_1) + self.A.dot(sigma_2) - l*self.r/R_max

            # iteratively build the Lagrangian value, accounting for multiple POVMs
            for i in range(self.N):
                # number of elements in the ith POVM
                Ni = self.N_list[i]
                # ith POVM in matrix form
                POVM_mat_i = self.POVM_mat_list[i]
                # number of repetitions of ith POVM measurement
                Ri = self.R_list[i]

                # the probability distributions corresponding to the POVM
                p_1_i = (POVM_mat_i.dot(sigma_1) + self.epsilon_o/Ni) / (1. + self.epsilon_o)
                p_2_i = (POVM_mat_i.dot(sigma_2) + self.epsilon_o/Ni) / (1. + self.epsilon_o)

                L_val = L_val - l * (Ri/R_max) * np.log(np.sqrt(p_1_i).dot(np.sqrt(p_2_i)))

            return L_val

        # gradient of the objective function
        def gradL(sigma_ds):
            sigma_1 = sigma_ds[0: self.n]
            sigma_2 = sigma_ds[self.n: 2*self.n]

            # gradient with respect to sigma_1: -A - \sum_i (\lambda R_i/2AffH^{i}) \sum_k \sqrt{p_2/p_1}^{i}_k E^{i}_k/(1 + \epsilon_o)
            # gradient with respect to sigma_2: A - \sum_i (\lambda R_i/2AffH^{i}) \sum_k \sqrt{p_1/p_2}^{i}_k E^{i}_k/(1 + \epsilon_o)

            # start with the terms that don't depend on POVMs
            grad_sigma_1 = -self.A
            grad_sigma_2 = self.A

            for i in range(self.N):
                # number of elements in the ith POVM
                Ni = self.N_list[i]
                # ith POVM in matrix form
                POVM_mat_i = self.POVM_mat_list[i]
                # number of repetitions of ith POVM measurement
                Ri = self.R_list[i]

                # the probability distributions corresponding to the POVM:
                # p_1^{i}(k) = (<E^{i}_k, sigma_1> + \epsilon_o/Nm)/(1 + \epsilon_o) and
                # p_2^{i}(k) = (<E^{i}_k, sigma_2> + \epsilon_o/Nm)/(1 + \epsilon_o)
                p_1_i = (POVM_mat_i.dot(sigma_1) + self.epsilon_o/Ni) / (1. + self.epsilon_o)
                p_2_i = (POVM_mat_i.dot(sigma_2) + self.epsilon_o/Ni) / (1. + self.epsilon_o)

                # Hellinger affinity between p_1 and p_2
                AffH_i = np.sqrt(p_1_i).dot(np.sqrt(p_2_i))

                # gradient with respect to sigma_1
                grad_sigma_1 = grad_sigma_1 - 0.5*l* (Ri/R_max) * np.sqrt(p_2_i/p_1_i).dot(POVM_mat_i)/(AffH_i * (1. + self.epsilon_o))

                # gradient with respect to sigma_2: A - (\lambda/2AffH) \sum_i \sqrt{p_1/p_2}_i E_i/(1 + \epsilon_o)
                # in matrix form, A - \lambda/2 \sqrt{p_1/p_2}^T POVM_mat/AffH
                grad_sigma_2 = grad_sigma_2 - 0.5*l* (Ri/R_max) * np.sqrt(p_1_i/p_2_i).dot(POVM_mat_i)/(AffH_i * (1. + self.epsilon_o))

            grad_sigma_ds = np.concatenate((grad_sigma_1, grad_sigma_2))

            return grad_sigma_ds

        # the other part of the objective function is an indicator function
        # because Nesterov's second method always has the interates in the domain, the indicator is always zero
        P = lambda sigma_ds: 0.

        # proximal operator of an indicator function is projector
        def prox_lP(sigma_ds, l, tol):
            # first ensure that we have a Hermitian matrix before projecting on the set of density matrices 
            # the projection must then be embedded into a real vector space

            sigma_1_projection = project_on_density_matrices_flattened(\
                                        embed_hermitian_matrix_real_vector_space(sigma_ds[0: self.n], reverse = True, flatten = True))

            sigma_2_projection = project_on_density_matrices_flattened(\
                                        embed_hermitian_matrix_real_vector_space(sigma_ds[self.n: 2*self.n], reverse = True, flatten = True))

            sigma_ds_projection_embedded = np.concatenate((embed_hermitian_matrix_real_vector_space(sigma_1_projection),\
                                                           embed_hermitian_matrix_real_vector_space(sigma_2_projection)))

            return sigma_ds_projection_embedded

        # perform the minimization using Nesterov's second method (accelerated proximal gradient)
        sigma_ds_opt = minimize_proximal_gradient_nesterov(Lagr, P, gradL, prox_lP, self.mldm_sigma_ds_o, tol = self.tol)

        # store the optimizal point as initial condition for future use
        self.mldm_sigma_ds_o = sigma_ds_opt

        return (sigma_ds_opt, Lagr(sigma_ds_opt))

    def find_density_matrices_saddle_point(self):
        """
            Solves the optimization problem

            \max_{sigma_1, sigma_2 \in X} {Tr(A sigma_1) - Tr(A sigma_2) | -R\log(\sum_i \sqrt{(p_1)_i (p_2)_i}) <= r}
                = - \min{sigma_1, sigma_2 \in X} {-Tr(A sigma_1) + Tr(A sigma_2) | -\log(\sum_i \sqrt{(p_1)_i (p_2)_i}) <= r/R}

            where (p_1)_i = (Tr(E_i sigma_1) + \epsilon_o/Nm) / (1 + \epsilon_o), (p_2)_i = (Tr(E_i sigma_2) + \epsilon_o/Nm) / (1 + \epsilon_o)
            and Nm denotes the number of POVM elements. X is the set of density matrices, A is the observable whose expectation value is calculated,
            and {E_i}_{i = 1}^{N_m} form a POVM. R, r > 0 are parameters, where 'R' denotes the number of observations of the POVM measurement.

            The density matrices sigma_1*, sigma_2* achieving this maximum/minimum correspond to the (x, y) component of the saddle point of \Phi_r.

            The problem is solved using duality. The Lagrangian is given as

            L(sigma_1, sigma_2; \lambda) = -Tr(A sigma_1) + Tr(A sigma_2) - \lambda (\log(\sum_i \sqrt{Tr(E_i sigma_1) Tr(E_i sigma_2)}) + r/R)

            where the constraint sigma_1, sigma_2 \in X is kept implicit, and \lambda is the dual variable. The dual function is then given as

            g(\lambda) = \min_{sigma_1, sigma_2 \in X} L(sigma_1, sigma_2; \lambda)

            and the dual problem is

            max_{\lambda >= 0} g(\lambda)

            Note that strong duality holds because for any sigma_1 = sigma_2 \in X, we have Tr(E_i sigma_1) = Tr(E_i sigma_2), so p_1 = p_2 and
            \sum_i \sqrt{(p_1)_i (p_2)_i} = \sum_i p_1 = 1, and therefore the constraint is strictly satisfied for r > 0,
            and so Slater's condition holds.

            If the dual optimum is achieved at \lambda*, then the optimal density matrices sigma_1*, sigma_2* are given by solving
            argmin_{sigma_1, sigma_2 \in X} L(sigma_1, sigma_2; \lambda*)

            Note that sigma_1*, sigma_2* need not be unique.
        """
        # negative of the dual function
        def minus_g(l):
            g_val = self.minimize_lagrangian_density_matrices_saddle_point(l)[1]

            return -g_val

        # perform the dual optimization
        l_opt = sp.optimize.minimize_scalar(minus_g, bounds = (0, 1e5), method = 'bounded').x

        # find the primal optimal using l*
        sigma_ds_opt = self.minimize_lagrangian_density_matrices_saddle_point(l_opt)[0]

        # obtain sigma_1_opt and sigma_2_opt
        self.sigma_1_opt = sigma_ds_opt[0: self.n]
        self.sigma_2_opt = sigma_ds_opt[self.n: 2*self.n]

        return (self.sigma_1_opt, self.sigma_2_opt)
    ###----- Finding x, y maximum of \Phi_r

    ###----- Finding alpha minimum of \Phi_r
    def maximize_Phi_r_density_matrices(self, phi_list, alpha):
        """
            Sovles the optimization problem
            
            \max_{sigma_1, sigma_2 \in X} \Phi_r(sigma_1, sigma_2; phi, alpha)
            = -\min_{sigma_1, sigma_2 \in X} -\Phi_r(sigma_1, sigma_2; phi, alpha)

            for a fixed vector phi \in R^{N_m} and a number alpha > 0.

            The objective function is given as

            \Phi_r(sigma_1, sigma_2; phi, alpha) = Tr(A sigma_1) - Tr(A sigma_2)
                                                    + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(-phi^{i}_k/alpha) (p^{i}_1)_k)
                                                        + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(phi^{i}_k/alpha) (p^{i}_2)_k)
                                                            + 2 alpha r

            where
            (p^{i}_1)_k = (Tr(E^(i)_k sigma_1) + \epsilon_o/Nm) / (1 + \epsilon_o) and
            (p^{i}_2)_k = (Tr(E^{i}_k sigma_2) + \epsilon_o/Nm) / (1 + \epsilon_o)
            are the probability distributions corresponding to the ith POVM {E^{i}_k}_{k = 1}^{N_i} with N_i elements.
            R_i > 0 is a parameter that denotes the number of observations of the ith type of measurement (i.e., ith POVM).
            There are a total of N POVMs.

            The optimization is performed using proximal gradient.

            Is is expected that all the Hermitian matrices are embedded into a real vector space.
            If any result do contain matrices (by converting back the embedded vector into a matrix), these matrices
            are expected to be flattened (in row-major style).
            Operations, where possible, are performed as matrix/array operations.
        """
        # for repeated use in functions
        phi_alpha_list = [phi/alpha for phi in phi_list]
        phi_alpha_min_list = [np.min(phi_alpha) for phi_alpha in phi_alpha_list]
        phi_alpha_max_list = [np.max(phi_alpha) for phi_alpha in phi_alpha_list]

        # we work with direct sum sigma_ds = (sigma_1, sigma_2) for use in pre-written algorithms
        # the objective function (we work with negative of \Phi_r so that we can minimize instead of maximize)
        def f(sigma_ds):
            sigma_1 = sigma_ds[0: self.n]
            sigma_2 = sigma_ds[self.n: 2*self.n]

            # start with the terms that don't depend on POVMs
            f_val = -self.A.dot(sigma_1) + self.A.dot(sigma_2) - 2.*alpha*self.r

            # iteratively build the function value, accounting for multiple POVMs
            for i in range(self.N):
                # number of elements in the ith POVM
                Ni = self.N_list[i]
                # ith POVM in matrix form
                POVM_mat_i = self.POVM_mat_list[i]
                # number of repetitions of ith POVM measurement
                Ri = self.R_list[i]
                # phi/alpha components and its maximum and minimum value corresponding to ith POVM
                phi_alpha_i = phi_alpha_list[i]
                phi_alpha_i_min = phi_alpha_min_list[i]
                phi_alpha_i_max = phi_alpha_max_list[i]

                # the probability distributions corresponding to the POVM:
                # p_1^{i}(k) = (<E^{i}_k, sigma_1> + \epsilon_o/Nm)/(1 + \epsilon_o) and
                # p_2^{i}(k) = (<E^{i}_k, sigma_2> + \epsilon_o/Nm)/(1 + \epsilon_o)
                p_1_i = (POVM_mat_i.dot(sigma_1) + self.epsilon_o/Ni) / (1. + self.epsilon_o)
                p_2_i = (POVM_mat_i.dot(sigma_2) + self.epsilon_o/Ni) / (1. + self.epsilon_o)

                f_val = f_val + alpha*Ri*(phi_alpha_i_min - phi_alpha_i_max)\
                                    - alpha*Ri*np.log(np.exp(phi_alpha_i_min - phi_alpha_i).dot(p_1_i))\
                                        - alpha*Ri*np.log(np.exp(phi_alpha_i - phi_alpha_i_max).dot(p_2_i))

            return f_val

        def gradf(sigma_ds):
            sigma_1 = sigma_ds[0: self.n]
            sigma_2 = sigma_ds[self.n: 2*self.n]

            # start with the terms that don't depend on POVMs
            # gradient with respect to sigma_1
            gradf_sigma_1_val = -self.A
            # gradient with respect to sigma_2
            gradf_sigma_2_val = self.A

            for i in range(self.N):
                # number of elements in the ith POVM
                Ni = self.N_list[i]
                # ith POVM in matrix form
                POVM_mat_i = self.POVM_mat_list[i]
                # number of repetitions of ith POVM measurement
                Ri = self.R_list[i]
                # phi/alpha components and its maximum and minimum value corresponding to ith POVM
                phi_alpha_i = phi_alpha_list[i]
                phi_alpha_i_min = phi_alpha_min_list[i]
                phi_alpha_i_max = phi_alpha_max_list[i]

                # the probability distributions corresponding to the POVM:
                # p_1^{i}(k) = (<E^{i}_k, sigma_1> + \epsilon_o/Nm)/(1 + \epsilon_o) and
                # p_2^{i}(k) = (<E^{i}_k, sigma_2> + \epsilon_o/Nm)/(1 + \epsilon_o)
                p_1_i = (POVM_mat_i.dot(sigma_1) + self.epsilon_o/Ni) / (1. + self.epsilon_o)
                p_2_i = (POVM_mat_i.dot(sigma_2) + self.epsilon_o/Ni) / (1. + self.epsilon_o)

                # gradient with respect to sigma_1
                gradf_sigma_1_val = gradf_sigma_1_val - alpha*Ri*np.exp(phi_alpha_i_min - phi_alpha_i).dot(POVM_mat_i) / \
                                                                            (np.exp(phi_alpha_i_min - phi_alpha_i).dot(p_1_i) * (1. + self.epsilon_o))

                # gradient with respect to sigma_2
                gradf_sigma_2_val = gradf_sigma_2_val - alpha*Ri*np.exp(phi_alpha_i - phi_alpha_i_max).dot(POVM_mat_i) / \
                                                                            (np.exp(phi_alpha_i - phi_alpha_i_max).dot(p_2_i) * (1. + self.epsilon_o))

            # gradient with respect to sigma_ds
            gradf_val = np.concatenate((gradf_sigma_1_val, gradf_sigma_2_val))

            return gradf_val

        # the other part of the objective function is an indicator function on X x X, so it is set to zero because all iterates in Nesterov's
        # second method are inside the domain
        P = lambda sigma_ds: 0.

        # proximal operator of an indicator function is a projection
        def prox_lP(sigma_ds, l, tol):
            # first ensure that we have a Hermitian matrix before projecting on the set of density matrices 
            # the projection must then be embedded into a real vector space

            sigma_1_projection = project_on_density_matrices_flattened(\
                                        embed_hermitian_matrix_real_vector_space(sigma_ds[0: self.n], reverse = True, flatten = True))

            sigma_2_projection = project_on_density_matrices_flattened(\
                                        embed_hermitian_matrix_real_vector_space(sigma_ds[self.n: 2*self.n], reverse = True, flatten = True))

            sigma_ds_projection_embedded = np.concatenate((embed_hermitian_matrix_real_vector_space(sigma_1_projection),\
                                                           embed_hermitian_matrix_real_vector_space(sigma_2_projection)))

            return sigma_ds_projection_embedded

        # perform the minimization using Nesterov's second method (accelerated proximal gradient)
        sigma_ds_opt = minimize_proximal_gradient_nesterov(f, P, gradf, prox_lP, self.mpdm_sigma_ds_o, tol = self.tol)

        # store the optimal point as initial condition for future use
        self.mpdm_sigma_ds_o = sigma_ds_opt

        # obtain the density matrices at the optimum
        sigma_1_opt = sigma_ds_opt[0: self.n]
        sigma_2_opt = sigma_ds_opt[self.n: 2*self.n]

        return (sigma_1_opt, sigma_2_opt, -f(sigma_ds_opt))

    def find_alpha_saddle_point(self, phi_alpha_opt_list, Phi_r_opt):
        """
            Solves the optimization problem

            \min_{alpha > 0} (bar{\Phi_r}((phi/alpha)* x alpha, alpha) - 2\Phi*(r))^2

            where (phi/alpha)* is the saddle point component of \Phi_r, and \Phi*(r) is the corresponding saddle point value.

            The function bar{\Phi_r} is given as

            bar{\Phi_r}(phi, alpha) = \max_{sigma_1, sigma_2 \in X} \Phi_r(sigma_1, sigma_2; phi, alpha)

            for any given vector phi \in R^{N_m} and alpha > 0. The function \Phi_r is given as

            \Phi_r(sigma_1, sigma_2; phi, alpha) = Tr(A sigma_1) - Tr(A sigma_2)
                                                    + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(-phi^{i}_k/alpha) (p^{i}_1)_k)
                                                        + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(phi^{i}_k/alpha) (p^{i}_2)_k)
                                                            + 2 alpha r
            where
            (p^{i}_1)_k = (Tr(E^(i)_k sigma_1) + \epsilon_o/Nm) / (1 + \epsilon_o) and
            (p^{i}_2)_k = (Tr(E^{i}_k sigma_2) + \epsilon_o/Nm) / (1 + \epsilon_o) 
            are the probability distributions corresponding to the ith POVM {E^{i}_k}_{k = 1}^{N_i} with N_i elements.
            R_i > 0 is a parameter that denotes the number of observations of the ith type of measurement (i.e., ith POVM).
            There are a total of N POVMs.
        """
        # consider \bar{\Phi_r}(phi*, alpha) as a function of alpha
        # we know that at the saddle point, \bar{\Phi_r}(phi*, alpha*) = 2\Phi*(r)
        # so minimize (\bar{\Phi_r}(phi*, alpha) - 2\Phi*(r))^2
        def Phi_r_bar_alpha(alpha):
            phi_list = [phi_alpha_opt * alpha for phi_alpha_opt in phi_alpha_opt_list]
            Phi_r_bar_alpha_val = (self.maximize_Phi_r_density_matrices(phi_list = phi_list, alpha = alpha)[2]\
                                                - 2.*Phi_r_opt)**2
            return Phi_r_bar_alpha_val

        # perform the minimization
        alpha_opt = sp.optimize.minimize_scalar(Phi_r_bar_alpha, bounds = (1e-16, 1e3), method = 'bounded').x

        return alpha_opt
    ###----- Finding alpha minimum of \Phi_r

    ###----- Finding the constant in the expectation value estimator
    def find_constant_expectation_value_estimator(self, sigma_1_opt, sigma_2_opt, phi_opt_list, alpha_opt):
        """
            Calculate the constant 'c' directly from the saddle point.
            Created mainly to prevent cluttering the namespace of the calling function.

            c = 0.5 \max_{sigma_1} [Tr(A sigma_1) + \sum_{i = 1}^N alpha* R_i log(\sum_{k = 1}^{N_i} exp(-phi*^{i}_k/alpha*) (p^{i}_1)_k)]
                 - 0.5 \max_{sigma_2} [-Tr(A sigma_2) + \sum_{i = 1}^N alpha* R_i log(\sum_{k = 1}^{N_i} exp(phi*^{i}_k/alpha*) (p^{i}_2)_k)]
        """
        # start with terms that don't depend on the POVMs
        c = 0.5*(self.A.dot(sigma_1_opt) + self.A.dot(sigma_2_opt))

        for i in range(self.N):
            # number of elements in the ith POVM
            Ni = self.N_list[i]
            # ith POVM in matrix form
            POVM_mat_i = self.POVM_mat_list[i]
            # number of repetitions of ith POVM measurement
            Ri = self.R_list[i]

            # the probability distributions corresponding to sigma_1*, sigma_2*:
            # p^{i}_1(k) = (<E^{i}_k, sigma_1*> + \epsilon_o/Ni)/(1 + \epsilon_o) and
            # p^{i}_2(k) = (<E^{i}_k, sigma_2*> + \epsilon_o/Ni)/(1 + \epsilon_o)
            p_1_i = (POVM_mat_i.dot(sigma_1_opt) + self.epsilon_o/Ni) / (1. + self.epsilon_o)
            p_2_i = (POVM_mat_i.dot(sigma_2_opt) + self.epsilon_o/Ni) / (1. + self.epsilon_o)

            phi_alpha_i = phi_opt_list[i]/alpha_opt
            phi_alpha_i_min, phi_alpha_i_max = np.min(phi_alpha_i), np.max(phi_alpha_i)

            c = c + 0.5*(-alpha_opt*Ri*(phi_alpha_i_max + phi_alpha_i_min)\
                            + alpha_opt*Ri*np.log(np.exp(phi_alpha_i_min - phi_alpha_i).dot(p_1_i))\
                                - alpha_opt*Ri*np.log(np.exp(phi_alpha_i - phi_alpha_i_max).dot(p_2_i)))

        # store the value of the constant
        self.c = c

        return c
    ###----- Finding the constant in the expectation value estimator

    ###----- Constructing the expectation value estimator
    def find_expectation_value_estimator(self):
        """
            Constructs an estimator for expectation value of the observable A with respect to the unknown state sigma.

            First, the saddle point sigma_1*, sigma_2*, phi*, alpha* of the function

            \Phi_r(sigma_1, sigma_2; phi, alpha) = Tr(A sigma_1) - Tr(A sigma_2)
                                                    + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(-phi^{i}_k/alpha) (p^{i}_1)_k)
                                                        + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(phi^{i}_k/alpha) (p^{i}_2)_k)
                                                            + 2 alpha r

            is found. Here,
            (p^{i}_1)_k = (Tr(E^(i)_k sigma_1) + \epsilon_o/Nm) / (1 + \epsilon_o) and
            (p^{i}_2)_k = (Tr(E^{i}_k sigma_2) + \epsilon_o/Nm) / (1 + \epsilon_o) 
            are the probability distributions corresponding to the ith POVM {E^{i}_k}_{k = 1}^{N_i} with N_i elements.
            R_i > 0 is a parameter that denotes the number of observations of the ith type of measurement (i.e., ith POVM).
            There are a total of N POVMs.

            Then, an estimator is constructed as follows.

            \hat{F}(\omega^{1}_1, ..., \omega^{1}_{R_1}, ..., \omega^{N}_1, ..., \omega^{N}_{R_N})
                    = \sum_{i = 1}^N \sum_{l = 1}^{R_i} phi*(\omega^{i}_l) + c

            where the constant 'c' is given by the optimization problem

            c = 0.5 \max_{sigma_1} [Tr(A sigma_1) + \sum_{i = 1}^N alpha* R_i log(\sum_{k = 1}^{N_i} exp(-phi*^{i}_k/alpha*) (p^{i}_1)_k)]
                 - 0.5 \max_{sigma_2} [-Tr(A sigma_2) + \sum_{i = 1}^N alpha* R_i log(\sum_{k = 1}^{N_i} exp(phi*^{i}_k/alpha*) (p^{i}_2)_k)]

            We use the convention that the ith POVM outcomes are labelled as \Omega_i = {0, ..., N_m - 1}, as Python is zero-indexed.
        """
        # find the x, y component of the saddle point
        sigma_1_opt, sigma_2_opt = self.find_density_matrices_saddle_point()

        # the saddle point value of \Phi_r
        Phi_r_opt = 0.5*self.A.dot(sigma_1_opt - sigma_2_opt)

        # construct (phi/alpha)* at saddle point using sigma_1* and sigma_2*
        phi_alpha_opt_list = [0]*self.N
        for i in range(self.N):
            # number of elements in the ith POVM
            Ni = self.N_list[i]
            # ith POVM in matrix form
            POVM_mat_i = self.POVM_mat_list[i]

            # the probability distributions corresponding to sigma_1*, sigma_2*:
            # p^{i}_1(k) = (<E^{i}_k, sigma_1*> + \epsilon_o/Ni)/(1 + \epsilon_o) and
            # p^{i}_2(k) = (<E^{i}_k, sigma_2*> + \epsilon_o/Ni)/(1 + \epsilon_o)
            p_1_i = (POVM_mat_i.dot(sigma_1_opt) + self.epsilon_o/Ni) / (1. + self.epsilon_o)
            p_2_i = (POVM_mat_i.dot(sigma_2_opt) + self.epsilon_o/Ni) / (1. + self.epsilon_o)

            # (phi/alpha)* at the saddle point
            phi_alpha_opt_list[i] = 0.5*np.log(p_1_i/p_2_i)

        # find the alpha component of the saddle point
        self.alpha_opt = self.find_alpha_saddle_point(phi_alpha_opt_list, Phi_r_opt)

        # obtain phi* at the saddle point
        self.phi_opt_list = [phi_alpha_opt * self.alpha_opt for phi_alpha_opt in phi_alpha_opt_list]

        # find the constant in the estimator
        c = self.find_constant_expectation_value_estimator(sigma_1_opt, sigma_2_opt, self.phi_opt_list, self.alpha_opt)

        # build the estimator
        def estimator(data_list):
            """
                Given Ri independent and identically distributed elements from \Omega_i = {1, .., n_i} sampled as per p_{A^{i}(sigma)} for
                i = 1, ..., N, gives the estimate for the expectation value <A>_sigma = Tr(A sigma).

                \hat{F}(\omega^{1}_1, ..., \omega^{1}_{R_1}, ... \omega^{N}_1, ..., \omega^{N}_{R_N})
                                    = \sum_{i = 1}^N \sum_{l = 1}^{R_i} phi^{i}*(\omega^{i}_l) + c
            """
            N = len(data_list)

            # start with the terms that don't depend on the POVMs
            estimate = c

            # build the estimate iteratively, accounting for each POVM
            for i in range(N):
                # phi* component at the saddle point corresponding to the ith POVM
                phi_opt_i = self.phi_opt_list[i]

                # data corresponding to the ith POVM
                data_i = data_list[i]

                estimate = estimate + np.sum([phi_opt_i[l] for l in data_i])

            return estimate

        self.estimator = estimator

        return (estimator, Phi_r_opt)
    ###----- Constructing the expectation value estimator
