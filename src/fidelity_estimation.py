"""
    Given a pure state rho, and an unknown state sigma, finds the fidelity F(rho, sigma) = Tr(rho sigma) between rho and sigma.

    Tomographic measurements of sigma using different N POVMs {E^{i}_k}_{k = 1}^{N_i} (1 <= i <= N) is used to estimate the fidelity.

    This is done by following Juditsky & Nemirovski's approach.
    First, we find the saddle point of the concave-convex function

    \Phi_r(sigma_1, sigma_2; phi, alpha) = Tr(rho sigma_1) - Tr(rho sigma_2)
                                                + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(-phi^{i}_k/alpha) (p^{i}_1)_k)
                                                    + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(phi^{i}_k/alpha) (p^{i}_2)_k)
                                                        + 2 alpha r

    where
    (p^{i}_1)_k = (Tr(E^(i)_k sigma_1) + \epsilon_o/Nm) / (1 + \epsilon_o) and
    (p^{i}_2)_k = (Tr(E^{i}_k sigma_2) + \epsilon_o/Nm) / (1 + \epsilon_o)
    are the probability distributions corresponding to the ith POVM {E^{i}_k}_{k = 1}^{N_i} with N_i elements.
    R_i > 0 is a parameter that denotes the number of observations of the ith type of measurement (i.e., ith POVM).
    There are a total of N POVMs.

    X is the set of density matrices, rho is the "target" density matrix. r > 0 is a parameter.

    Then, given the saddle point sigma_1*, sigma_2*, phi*, alpha*, we can construct an estimator
    \hat{F}(\omega^{1}_1, ..., \omega^{1}_{R_1}, ... \omega^{N}_1, ..., \omega^{N}_{R_N})
                        = \sum_{i = 1}^N \sum_{l = 1}^{R_i} phi^{i}*(\omega^{i}_l) + c

    where the constant 'c' is given by the optimization problem

    c = 0.5 \max_{sigma_1} [Tr(rho sigma_1) + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(-phi^{i}_k/alpha) (p^{i}_1)_k)]
         - 0.5 \max_{sigma_2} [-Tr(rho sigma_2) + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(phi^{i}_k/alpha) (p^{i}_2)_k)]

    The saddle point value \Phi*(r) gives an upper bound for the confidence interval within which the error lies.

    Author: Akshay Seshadri
"""

import warnings

import numpy as np
import scipy as sp
from scipy import optimize

import project_root # noqa
from src.optimization.project_density_matrices_set import project_on_density_matrices_flattened
from src.optimization.proximal_gradient import minimize_proximal_gradient_nesterov
from src.utilities.qi_utilities import embed_hermitian_matrix_real_vector_space, generate_random_state

class Fidelity_Estimation_Manager():
    """
        Solves the different optimization problems required for fidelity estimation using Juditsky & Nemirovski's approach.

        This involves finding a saddle point of the function

        \Phi_r(sigma_1, sigma_2; phi, alpha) = Tr(rho sigma_1) - Tr(rho sigma_2)
                                                + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(-phi^{i}_k/alpha) (p^{i}_1)_k)
                                                    + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(phi^{i}_k/alpha) (p^{i}_2)_k)
                                                        + 2 alpha r

        where
        (p^{i}_1)_k = (Tr(E^(i)_k sigma_1) + \epsilon_o/Nm) / (1 + \epsilon_o) and
        (p^{i}_2)_k = (Tr(E^{i}_k sigma_2) + \epsilon_o/Nm) / (1 + \epsilon_o)
        are the probability distributions corresponding to the ith POVM {E^{i}_k}_{k = 1}^{N_i} with N_i elements.
        R_i > 0 is a parameter that denotes the number of observations of the ith type of measurement (i.e., ith POVM).
        There are a total of N POVMs.

        X is the set of density matrices, rho is the "target" density matrix. r > 0 is a parameter.

        Then, given the saddle point sigma_1*, sigma_2*, phi*, alpha*, we can construct an estimator
        \hat{F}(\omega^{1}_1, ..., \omega^{1}_{R_1}, ... \omega^{N}_1, ..., \omega^{N}_{R_N})
                            = \sum_{i = 1}^N \sum_{l = 1}^{R_i} phi^{i}*(\omega^{i}_l) + c

        where the constant 'c' is given by the optimization problem

        c = 0.5 \max_{sigma_1} [Tr(rho sigma_1) + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(-phi^{i}_k/alpha) (p^{i}_1)_k)]
             - 0.5 \max_{sigma_2} [-Tr(rho sigma_2) + \sum_{i = 1}^N alpha R_i log(\sum_{k = 1}^{N_i} exp(phi^{i}_k/alpha) (p^{i}_2)_k)]

        The saddle point value \Phi*(r) gives an upper bound for the confidence interval within which the error lies.
    """
    def __init__(self, R_list, epsilon, rho, POVM_list, epsilon_o, tol = 1e-6, random_init = False, print_progress = True):
        """
            Assigns values to parameters and defines and initializes functions.

            We perform most of the optimization by isometrically embedding Hermitian matrices into a real vector space.
            So, we refer to the matrix form by appending '_full' to the variable name.

            Arguments:
                - R_list         : list of repetitions used for each POVM
                - epsilon        : 1 - confidence level, should be between 0 and 0.25, end points excluded
                - rho            : target state; must be a pure state density matrix
                - POVM_list      : list of POVMs to be measured
                - epsilon_o      : constant to prevent zero probabilities in Born's rule
                - tol            : tolerance used by the optimization algorithms
                - random_init    : if True, a random initial condition is used for the optimization
                - print_progress : if True, the progress of optimization is printed

            All the matrices are expected to be flattened in row-major style.
        """
        # confidence level
        self.epsilon = epsilon

        # obtain 'r' from \epsilon
        self.r = np.log(2./epsilon)

        # constant to keep the probabilities in Born rule positive
        self.epsilon_o = epsilon_o

        # density matrix of the system (before embedding)
        self.rho_full = rho

        # list of POVMs, one corresponding to each type of measurement
        self.POVM_list_full = POVM_list

        # dimension of the density matrix (full state)
        self.n = rho.size
        # number of POVMs (also the number of types of measurement)
        self.N = len(POVM_list)
        # number of elements in each POVM
        self.N_list = [len(POVM) for POVM in POVM_list]

        # list of number of repetitions of each (type of) measurement
        # convert R to a floating point number if it already isn't one
        if type(R_list) not in [list, tuple, np.ndarray]:
            self.R_list = [float(R_list)]*self.N
        else:
            self.R_list = [float(R) for R in R_list]

        # embed all Hermitian matrices into a real vector space
        # size of rho before embedding is n^2 (flattened, over complex vector space) and after embedding is also n^2 (but over a real vector space)
        self.rho = embed_hermitian_matrix_real_vector_space(rho)
        self.POVM_list = [[embed_hermitian_matrix_real_vector_space(E_i) for E_i in POVM] for POVM in POVM_list]

        # convert each (embedded) POVM into a matrix
        self.POVM_mat_list = [np.vstack(POVM) for POVM in self.POVM_list]

        # tolerance for all the computations
        self.tol = tol

        # create a state for initializing minimize_lagrangian_density_matrices (to be used specifically for find_density_matrices_saddle_point)
        if not random_init:
            # use a slightly perturbed target state as initial condition
            sigma_init = 0.9 * self.rho_full + 0.1 * (np.eye(int(np.round(np.sqrt(self.n), decimals = 0)), dtype = 'complex128').ravel() - self.rho_full)
            sigma_init = embed_hermitian_matrix_real_vector_space(sigma_init)
        else:
            sigma_init = generate_random_state(int(np.round(np.sqrt(self.n), decimals = 0)), pure = False, density_matrix = True, flatten = True, random_seed = None)
            sigma_init = embed_hermitian_matrix_real_vector_space(sigma_init.astype('complex128'))

        # initialization for maximize_Phi_r_alpha_density_matrices (to be used specifically for find_density_matrices_alpha_saddle_point)
        self.mpdm_sigma_ds_o = np.concatenate((sigma_init, sigma_init))

        # determine whether to print progress
        self.print_progress = print_progress

        # counter to track progress
        if print_progress:
            self.progress_counter = 0

        # determine whether the optimization achieved the tolerance
        self.success = True

    ###----- Finding x, y maximum and alpha minimum of \Phi_r
    def maximize_Phi_r_alpha_density_matrices(self, alpha):
        """
            Solves the optimization problem

            \max_{sigma_1, sigma_2 \in X} \Phi_r_alpha(sigma_1, sigma_2)
            = -\min_{sigma_1, sigma_2 \in X} -\Phi_r_alpha(sigma_1, sigma_2)

            for a number alpha > 0.

            The objective function is given as

            Phi_r_alpha(sigma_1, sigma_2) = Tr(A sigma_1) - Tr(A sigma_2)
                                                + 2 alpha \sum_{i = 1}^N R_i log(\sum_{k = 1}^{N_i} \sqrt{(p^{i}_1)_k (p^{i}_2)_k})

            where
            (p^{i}_1)_k = (Tr(E^(i)_k sigma_1) + \epsilon_o/Nm) / (1 + \epsilon_o) and
            (p^{i}_2)_k = (Tr(E^{i}_k sigma_2) + \epsilon_o/Nm) / (1 + \epsilon_o)
            are the probability distributions corresponding to the ith POVM {E^{i}_k}_{k = 1}^{N_i} with N_i elements.
            R_i > 0 is a parameter that denotes the number of observations of the ith type of measurement (i.e., ith POVM).
            There are a total of N POVMs.

            The optimization is performed using accelerated proximal gradient (Nesterov's second method).

            Is is expected that all the Hermitian matrices are embedded into a real vector space.
            If any result do contain matrices (by converting back the embedded vector into a matrix), these matrices
            are expected to be flattened (in row-major style).
            Operations, where possible, are performed as matrix/array operations.
        """
        # we work with direct sum sigma_ds = (sigma_1, sigma_2) for use in pre-written algorithms
        # the objective function (we work with negative of \Phi_r_alpha so that we can minimize instead of maximize)
        def f(sigma_ds):
            sigma_1 = sigma_ds[0: self.n]
            sigma_2 = sigma_ds[self.n: 2*self.n]

            # start with the terms that don't depend on POVMs
            f_val = -self.rho.dot(sigma_1) + self.rho.dot(sigma_2)

            POVM_mats = np.array( self.POVM_mat_list )

            Ns = np.expand_dims( np.array( self.N_list ), 1 )
            Rs = np.array( self.R_list )

            # the probability distributions corresponding to the POVM:
            # p_1^{i}(k) = (<E^{i}_k, sigma_1> + \epsilon_o/Nm)/(1 + \epsilon_o) and
            # p_2^{i}(k) = (<E^{i}_k, sigma_2> + \epsilon_o/Nm)/(1 + \epsilon_o)
            p_1 = (POVM_mats.dot(sigma_1) + self.epsilon_o / Ns) / (1. + self.epsilon_o)
            p_2 = (POVM_mats.dot(sigma_2) + self.epsilon_o / Ns) / (1. + self.epsilon_o)

            return f_val - 2 * alpha * np.sum( Rs * np.log( np.sum( np.sqrt(p_1 * p_2), axis=1 ) ) ) 

        def gradf(sigma_ds):
            sigma_1 = sigma_ds[0: self.n]
            sigma_2 = sigma_ds[self.n: 2*self.n]

            # start with the terms that don't depend on POVMs
            # gradient with respect to sigma_1
            gradf_sigma_1_val = -self.rho
            # gradient with respect to sigma_2
            gradf_sigma_2_val = self.rho

            POVM_mats = np.array( self.POVM_mat_list )
            Ns = np.expand_dims( np.array( self.N_list ), 1 )
            Rs = np.expand_dims( np.array( self.R_list ), 1 )

            # the probability distributions corresponding to the POVM:
            # p_1^{i}(k) = (<E^{i}_k, sigma_1> + \epsilon_o/Nm)/(1 + \epsilon_o) and
            # p_2^{i}(k) = (<E^{i}_k, sigma_2> + \epsilon_o/Nm)/(1 + \epsilon_o)    
            p_1 = (POVM_mats.dot(sigma_1) + self.epsilon_o / Ns) / (1. + self.epsilon_o)
            p_2 = (POVM_mats.dot(sigma_2) + self.epsilon_o / Ns) / (1. + self.epsilon_o)

            AffHs = np.expand_dims( np.sum( np.sqrt(p_1) * np.sqrt(p_2), axis=1 ), 1 )

            gradf_sigma_1_val = gradf_sigma_1_val - alpha * np.sum( Rs * np.sum( np.expand_dims( np.sqrt(p_2/p_1), 2 ) * POVM_mats, axis=1 ) / (AffHs * (1. + self.epsilon_o)), axis=0 )

            gradf_sigma_2_val = gradf_sigma_2_val - alpha * np.sum( Rs * np.sum( np.expand_dims( np.sqrt(p_1/p_2), 2 ) * POVM_mats, axis=1 ) / (AffHs * (1. + self.epsilon_o)), axis=0 )

            # gradient with respect to sigma_ds
            return np.concatenate((gradf_sigma_1_val, gradf_sigma_2_val))

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
        sigma_ds_opt, error = minimize_proximal_gradient_nesterov(f, P, gradf, prox_lP, self.mpdm_sigma_ds_o, tol = self.tol, return_error = True)

        # check if tolerance is satisfied
        if error > self.tol:
            self.success = False
            warnings.warn("The tolerance for the optimization was not achieved. The estimates may be unreliable. Consider using a random initial condition by setting random_init = True.", MinimaxOptimizationWarning)

        # store the optimal point as initial condition for future use
        self.mpdm_sigma_ds_o = sigma_ds_opt

        # obtain the density matrices at the optimum
        self.sigma_1_opt = sigma_ds_opt[0: self.n]
        self.sigma_2_opt = sigma_ds_opt[self.n: 2*self.n]

        return (self.sigma_1_opt, self.sigma_2_opt, -f(sigma_ds_opt))

    def find_density_matrices_alpha_saddle_point(self):
        """
            Solves the optimization problem

            \min_{alpha > 0} (alpha r + 0.5*inf_phi bar{Phi_r}(phi, alpha))

            The function bar{\Phi_r} is given as

            bar{\Phi_r}(phi, alpha) = \max_{sigma_1, sigma_2 \in X} \Phi_r(sigma_1, sigma_2; phi, alpha)

            for any given vector phi \in R^{N_m} and alpha > 0.

            The infinum over phi of bar{\Phi_r} can be solved to obtain

            Phi_r_bar_alpha = \inf_phi bar{Phi_r}(phi, alpha)
                            = \max_{sigma_1, sigma_2 \in X} \inf_phi \Phi_r(sigma_1, sigma_2; phi, alpha)
                            = \max_{sigma_1, sigma_2 \in X} [Tr(rho sigma_1) - Tr(rho sigma_2)
                                + 2 alpha \sum_{i = 1}^N R_i log(\sum_{k = 1}^{N_i} \sqrt{(p^{i}_1)_k (p^{i}_2)_k})]
            where
            (p^{i}_1)_k = (Tr(E^(i)_k sigma_1) + \epsilon_o/Nm) / (1 + \epsilon_o) and
            (p^{i}_2)_k = (Tr(E^{i}_k sigma_2) + \epsilon_o/Nm) / (1 + \epsilon_o) 
            are the probability distributions corresponding to the ith POVM {E^{i}_k}_{k = 1}^{N_i} with N_i elements.
            R_i > 0 is a parameter that denotes the number of observations of the ith type of measurement (i.e., ith POVM).
            There are a total of N POVMs.

            We define

            Phi_r_alpha(sigma_1, sigma_2) = Tr(rho sigma_1) - Tr(rho sigma_2)
                                                + 2 alpha \sum_{i = 1}^N R_i log(\sum_{k = 1}^{N_i} \sqrt{(p^{i}_1)_k (p^{i}_2)_k})
            
            so that

            Phi_r_bar_alpha = \max_{sigma_1, sigma_2 \in X} Phi_r_alpha(sigma_1, sigma_2)

            Note that Phi_r_bar_alpha >= 0 since Phi_r_alpha(sigma_1, sigma_1) = 0.
        """
        def Phi_r_bar_alpha(alpha):
            # print progress to update the user, if required
            if self.print_progress:
                print("Finding the saddle-point (count: %s, max-count ~ 500)".ljust(60) %(self.progress_counter,), end = "\r", flush = True)

                # increment the counter
                self.progress_counter += 1

            Phi_r_bar_alpha_val = alpha*self.r + 0.5*self.maximize_Phi_r_alpha_density_matrices(alpha = alpha)[2]

            return Phi_r_bar_alpha_val

        # perform the minimization
        alpha_optimization_result = sp.optimize.minimize_scalar(Phi_r_bar_alpha, bounds = (1e-16, 1e3), method = 'bounded')

        # value of alpha at optimum
        self.alpha_opt = alpha_optimization_result.x

        # value of objective function at optimum: gives the risk
        self.Phi_r_bar_alpha_opt = alpha_optimization_result.fun

        if self.print_progress:
            print("Optimization complete.".ljust(60))

        # check if alpha optimization was successful
        if not alpha_optimization_result.success:
            self.success = False
            warnings.warn("The optimization has not converge properly to the saddle-point. The estimates may be unreliable. Consider using a random initial condition by setting random_init = True.", MinimaxOptimizationWarning)

        return (self.sigma_1_opt, self.sigma_2_opt, self.alpha_opt)
    ###----- Finding x, y maximum and alpha minimum of \Phi_r

    ###----- Constructing the fidelity estimator
    def find_fidelity_estimator(self):
        """
            Constructs an estimator for fidelity between a pure state rho and an unknown state sigma.

            First, the saddle point sigma_1*, sigma_2*, phi*, alpha* of the function

            \Phi_r(sigma_1, sigma_2; phi, alpha) = Tr(rho sigma_1) - Tr(rho sigma_2)
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

            c = 0.5 \max_{sigma_1} [Tr(rho sigma_1) + \sum_{i = 1}^N alpha* R_i log(\sum_{k = 1}^{N_i} exp(-phi*^{i}_k/alpha*) (p^{i}_1)_k)]
                 - 0.5 \max_{sigma_2} [-Tr(rho sigma_2) + \sum_{i = 1}^N alpha* R_i log(\sum_{k = 1}^{N_i} exp(phi*^{i}_k/alpha*) (p^{i}_2)_k)]

            We use the convention that the ith POVM outcomes are labelled as \Omega_i = {0, ..., N_m - 1}, as Python is zero-indexed.
        """
        # find x, y, and alpha components of the saddle point
        sigma_1_opt, sigma_2_opt, alpha_opt = self.find_density_matrices_alpha_saddle_point()

        # the saddle point value of \Phi_r
        Phi_r_opt = self.Phi_r_bar_alpha_opt

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

        # obtain phi* at the saddle point
        self.phi_opt_list = [phi_alpha_opt * alpha_opt for phi_alpha_opt in phi_alpha_opt_list]

        # find the constant in the estimator
        self.c = 0.5*(self.rho.dot(sigma_1_opt) + self.rho.dot(sigma_2_opt))

        # build the estimator
        def estimator(data_list):
            """
                Given Ri independent and identically distributed elements from \Omega_i = {1, .., n_i} sampled as per p_{A^{i}(sigma)} for
                i = 1, ..., N, gives the estimate for the fidelity F(rho, sigma) = Tr(rho sigma).

                \hat{F}(\omega^{1}_1, ..., \omega^{1}_{R_1}, ... \omega^{N}_1, ..., \omega^{N}_{R_N})
                                    = \sum_{i = 1}^N \sum_{l = 1}^{R_i} phi^{i}*(\omega^{i}_l) + c
            """
            N = len(data_list)

            # start with the terms that don't depend on the POVMs
            estimate = self.c

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
    ###----- Constructing the fidelity estimator
