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
import cvxpy as cp

import project_root # noqa
from src.optimization.project_density_matrices_set import project_on_density_matrices_flattened
from src.optimization.proximal_gradient import minimize_proximal_gradient_nesterov
from src.utilities.qi_utilities import embed_hermitian_matrix_real_vector_space, generate_random_state

class Fidelity_Estimation_Manager_CVXPY():
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

        Equivalently, we can solve the convex optimization problem 

        Maximize_{sigma_1, sigma_2 \in X} 0.5 * Tr( rho (sigma_1 - sigma_2) )

        Subject to ln( AffH( A(sigma_1), A(sigma_2) ) ) >= ln( epsilon / 2 )

        where AffH is the Hellinger affinity between distributions A(sigma_1) and A(sigma_2) is a log-concave function defined by
        ln( AffH( A(x), A(y) ) = \sum_{i = 1}^N R_i log(\sum_{k = 1}^{N_i} \sqrt{(p^{i}_1)_k (p^{i}_2)_k})]

        This produces the same optimal values sigma_1* and sigma_2*, and the value alpha* is the optimal dual variable of this constraint.
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
        if tol != 1e-6:
            warnings.warn("Tolerances in CVXPY optimization are solver dependent. The `tol` parameter will be unused.")

        # create a state for initializing minimize_lagrangian_density_matrices (to be used specifically for find_density_matrices_saddle_point)
        if not random_init:
            warnings.warn("The initial condition for the necessary CVXPY solvers are fixed. The `random_init` parameter will be unused.")

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
        self.success = None

        # stores the compiled cvxpy problem
        self.cvxpy_prob = None

    def define_cvxpy_problem( self ):
        """
        Defines/compiles the convex optimization problem with cvxpy

            \max_{x, y \in X} <rho, x - y>

            \subject to ln( AffH( A(x), A(y) ) ) >= ln( epsilon / 2 )

        This only needs to be done once, and can then be called multiple times 
        """
        nq = int(np.log2(np.sqrt(self.n)))

        # Define variables
        x = cp.Variable((2**nq, 2**nq), hermitian=True, name='x')
        y = cp.Variable((2**nq, 2**nq), hermitian=True, name='y')

        # Establish constraints for optimization
        rho = cp.Constant( self.rho_full.reshape( 2**nq, 2**nq ) )
        
        epsilon_o = cp.Constant( self.epsilon_o )
        rhs = cp.Constant( np.log( self.epsilon / 2 ) )

        POVM_mats = [[cp.Constant(E_i.reshape( 2**nq, 2**nq )) for E_i in POVM] for POVM in self.POVM_list_full]
        
        Ns = cp.Constant( np.array([len(POVM) for POVM in POVM_mats]) )
        Rs = cp.Parameter( len( POVM_mats ), nonneg=True, name='Rs' )

        lnAffH = cp.Constant( 0 )
        for i in range( len( POVM_mats ) ):
            inner_sum = cp.Constant( 0 )
            for j in range( len( POVM_mats[i] ) ):
                p_1 = (cp.real( cp.trace( POVM_mats[i][j] @ x ) ) + epsilon_o / Ns[i]) / (1 + epsilon_o)
                p_2 = (cp.real( cp.trace( POVM_mats[i][j] @ y ) ) + epsilon_o / Ns[i]) / (1 + epsilon_o)

                inner_sum = inner_sum + cp.geo_mean( cp.vstack([p_1, p_2]) )

            lnAffH = lnAffH + Rs[i] * cp.log( inner_sum )

        cp_con = [lnAffH >= rhs, x >> 0, y >> 0, cp.trace( x ) == 1, cp.trace( y ) == 1]
        
        obj = 0.5 * cp.Maximize( cp.real( cp.trace( rho @ ( x - y ) ) ) )

        self.cvxpy_prob = cp.Problem( obj, cp_con )

    ###----- Finding x, y maximum and alpha minimum of \Phi_r
    def maximize_risk_density_matrices( self, R_list = None, solver='SCS' ):
        """
            Solves the convex optimization problem with cvxpy. 
            The solver SCS is packaged with cvxpy, but problem is compatible with MOSEK,
            which may be faster in practice. 

            \max_{x, y \in X} <rho, x - y>

            \subject to ln( AffH( A(x), A(y) ) ) >= ln( epsilon / 2 )

            The function ln( AffH( A(x), A(y) ) ) is given as

            ln( AffH( A(x), A(y) ) = \sum_{i = 1}^N R_i log(\sum_{k = 1}^{N_i} \sqrt{(p^{i}_1)_k (p^{i}_2)_k})]
            where
            (p^{i}_1)_k = (Tr(E^(i)_k sigma_1) + \epsilon_o/Nm) / (1 + \epsilon_o) and
            (p^{i}_2)_k = (Tr(E^{i}_k sigma_2) + \epsilon_o/Nm) / (1 + \epsilon_o) 
            are the probability distributions corresponding to the ith POVM {E^{i}_k}_{k = 1}^{N_i} with N_i elements.
            
            R_i > 0 is a parameter that denotes the number of observations of the ith type of measurement (i.e., ith POVM).
            There are a total of N POVMs.

            In relation to the saddle point problem, the optimal dual variable of our constraint is equivalent to alpha*
        """

        # Only need to compile the problem once
        if self.cvxpy_prob is None:
            self.define_cvxpy_problem()
            
        # Can use originally defined shot counts
        if R_list is not None:
            self.R_list = R_list
        
        self.cvxpy_prob.param_dict['Rs'].value = np.array( self.R_list )
        self.cvxpy_prob.solve( verbose=self.print_progress, warm_start=True, solver=solver )

        self.sigma_1_opt = embed_hermitian_matrix_real_vector_space(self.cvxpy_prob.var_dict['x'].value)
        self.sigma_2_opt = embed_hermitian_matrix_real_vector_space(self.cvxpy_prob.var_dict['y'].value)
        self.alpha_opt = self.cvxpy_prob.constraints[0].dual_value
        self.Phi_r_bar_alpha_opt = self.cvxpy_prob.value

        # check if alpha optimization was successful
        if self.cvxpy_prob.status in ['infeasible', 'unbounded']:
            self.success = False
            warnings.warn("The CVXPY optimization has not converge properly to the optimal. The estimates may be unreliable. Consider using a different solver.")
        self.success = True

        return (self.sigma_1_opt, self.sigma_2_opt, self.alpha_opt)

    ###----- Constructing the fidelity estimator
    def find_fidelity_estimator(self, solver='SCS'):
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
        sigma_1_opt, sigma_2_opt, alpha_opt = self.maximize_risk_density_matrices(solver=solver)

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
