"""
    Creates a fidelity estimator for any pure state, using randomized Pauli measurement strategy.
    
    Author: Akshay Seshadri
"""

import warnings

import numpy as np
import scipy as sp
from scipy import optimize

import project_root # noqa
from src.optimization.proximal_gradient import minimize_proximal_gradient_nesterov
from src.utilities.qi_utilities import generate_random_state, generate_special_state, generate_Pauli_operator, generate_POVM, embed_hermitian_matrix_real_vector_space
from src.utilities.noise_process import depolarizing_channel
from src.utilities.quantum_measurements import Measurement_Manager
from src.fidelity_estimation import Fidelity_Estimation_Manager

def project_on_box(v, l, u):
    """
        Projects the point v \in R^n on to the box C = {x \in R^n | l <= x <= u}, where the inequality x >= l and x <= u are to be interpreted
        componentwise (i.e., x_k >= l_k and x_k <= u_k).

        The projection of v on to the box is given as
        \Pi(v)_k = l_k if v_k <= l_k
                   v_k if l_k <= v_k <= u_k
                   u_k if v_k >= u_k

        Note that the above can be expressed in a compact form as \Pi(v)_k = min(max(v_k, l_k), u_k)

        Here, l_k and u_k can be -\infty or \infty respectively.
    """
    Pi_v = np.minimum(np.maximum(v, l), u)

    return Pi_v

class Pauli_Sampler_Fidelity_Estimation_Manager():
    """
        Computes the Juditsky & Nemirovski estimator and risk for pure target states when measurements
        are performed as per the randomized Pauli measurement strategy described in Box II.1 of PRA submission.

        In general, this involves finding a saddle point of the function

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

        The above procedure described can be expensive in large dimensions. For the case of randomized Pauli measurement (RPM) strategy,
        the algorithms are specialized so that very large dimensions can be handled.

        For arbitrary pure target states, the RPM strategy corresponds to randomly sampling Pauli operators according to some predetermined
        sampling probability, measuring these Pauli operators, and recording their outcomes (+1 or -1 eigenavalue).
        For stabilizer states, this measurement strategy reduces to uniformly randomly sampling from the stabilizer group (all elements
        except the identity) and measuring them.
    """
    def __init__(self, n, R, NF, epsilon, epsilon_o, tol = 1e-6, random_init = False, print_progress = True):
        """
            Assigns values to parameters and defines and initializes functions.

            The estimator depends on the dimension of the target state, the number of repetitions of the measurement, a normalization factor,
            and the confidence level.
            It is independent of the actual target state used for the RPM strategy, except through the normalization factor described below.

            The small parameter epsilon_o required to formalize Juditsky & Nemirovski's approach is used only in the optimization for finding alpha.
            It is not used in finding the optimal sigma_1 and sigma_2 because those are computed "by hand".

            Arguments:
                - n              : dimension of the system
                - R              : total number of repetitions used
                - NF             : the normalization factor, NF = \sum_i |tr(W_i rho)|,
                                   where the sum is over all non-identity Paulis and rho is the target state
                - epsilon        : 1 - confidence level, should be between 0 and 0.25, end points excluded
                - epsilon_o      : constant to prevent zero probabilities in Born's rule
                - tol            : tolerance used by the optimization algorithms
                - random_init    : if True, a random initial condition is used for the optimization
                - print_progress : if True, the progress of optimization is printed
        """
        # confidence level
        self.epsilon = epsilon

        # obtain 'r' from \epsilon
        self.r = np.log(2./epsilon)

        # constant to keep the probabilities in Born rule positive
        self.epsilon_o = epsilon_o

        # dimension of the system
        self.n = n

        # number of repetitions of the (minimax optimal) measurement
        self.R = R

        # the normalization factor, NF = \sum_i |tr(W_i rho)|; state dependent
        self.NF = NF

        # quantities defining the POVM
        self.omega1 = 0.5 * (n + NF - 1) / NF
        self.omega2 = 0.5 * (NF - 1) / NF

        # lower bound for the (classical) fidelity, used in the theory for optimization
        self.gamma = (epsilon/2)**(2/R)

        # minimum number of repetitions required for a risk less than 0.5
        self.Ro = np.ceil(np.log(2/epsilon) / np.abs(np.log(np.sqrt(self.omega1 * self.omega2) + np.sqrt(np.abs((1 - self.omega1) * (1 - self.omega2))))))

        # if gamma is not large enough, we have a risk of 0.5
        if R <= self.Ro:
            warnings.warn("The number of repetitions are very low. Consider raising the number of repetitions to at least %d." %self.Ro, MinimaxOptimizationWarning)

        # tolerance for all the computations
        self.tol = tol

        # initialization for maximize_Phi_r_density_matrices_multiple_measurements (to be used specifically for find_alpha_saddle_point_fidelity_estimation)
        if not random_init:
            # we choose lambda_1 = lambda_2 = 0.9, which corresponds to sigma_1 = 0.9 rho + 0.1 rho_1_perp, sigma_2 = 0.9 rho + 0.1 rho_2_perp
            self.mpdm_lambda_ds_o = np.array([0.9, 0.9])
        else:
            # take lambda_1 and lambda_2 as some random number between 0 and 1
            self.mpdm_lambda_ds_o = np.random.random(size = 2)

        # determine whether to print progress
        self.print_progress = print_progress

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

            Phi_r_alpha(sigma_1, sigma_2) = Tr(rho sigma_1) - Tr(rho sigma_2)
                                                + 2 alpha \sum_{i = 1}^N R_i log(\sum_{k = 1}^{N_i} \sqrt{(p^{i}_1)_k (p^{i}_2)_k})

            where
            (p^{i}_1)_k = (Tr(E^(i)_k sigma_1) + \epsilon_o/Nm) / (1 + \epsilon_o) and
            (p^{i}_2)_k = (Tr(E^{i}_k sigma_2) + \epsilon_o/Nm) / (1 + \epsilon_o)
            are the probability distributions corresponding to the ith POVM {E^{i}_k}_{k = 1}^{N_i} with N_i elements.
            R_i > 0 is a parameter that denotes the number of observations of the ith type of measurement (i.e., ith POVM).
            There are a total of N POVMs.

            We parametrize the density matrices as the following convex combination
                sigma_1 = lambda_1 rho + (1 - lambda_1) rho_1_perp
                sigma_2 = lambda_2 rho + (1 - lambda_2) rho_2_perp
            where 0 <= lambda_1, lambda_2 <= 1, and rho_1_perp and rho_2_perp are density matrices in the orthogonal complement of the target state rho.

            The minimax measurement strategy consists of a single POVM with two elements {Omega, Delta_Omega}. With respect to this POVM, the Born probabilities are
            Tr(Omega sigma_1)       = omega_1 lambda_1 + omega_2 (1 - lambda_1)
            Tr(Delta_Omega sigma_1) = (1 - omega_1) lambda_1 + (1 - omega_2) (1 - lambda_1)
            and a similar expression can be written for sigma_2.
            We include the parameter epsilon_o in the Born probabilities to avoid zero-division while calculating the derivative of Phi_r.

            Using the above, we reduce the optimization to two dimensions, irrespective of the dimension of rho.
            The optimization is performed using proximal gradient.
        """
        # we work with direct sum lambda_ds = (lambda_1, lambda_2) for use in pre-written algorithms
        # the objective function (we work with negative of \Phi_r_alpha so that we can minimize instead of maximize)
        def f(lambda_ds):
            lambda_1 = lambda_ds[0]
            lambda_2 = lambda_ds[1]

            # start with the terms that don't depend on POVMs
            f_val = -lambda_1 + lambda_2

            # number of repetitions of the POVM measurement
            R = self.R

            # the probability distributions corresponding to the minimax optimal POVM:
            # p_1^{i}(k) = (<E^{i}_k, sigma_1> + \epsilon_o/Ni)/(1 + \epsilon_o) and
            # p_2^{i}(k) = (<E^{i}_k, sigma_2> + \epsilon_o/Ni)/(1 + \epsilon_o)
            p_1 = (np.array([self.omega1 * lambda_1 + self.omega2 * (1 - lambda_1), (1 - self.omega1) * lambda_1 + (1 - self.omega2) * (1 - lambda_1)]) + self.epsilon_o/2) / (1. + self.epsilon_o)
            p_2 = (np.array([self.omega1 * lambda_2 + self.omega2 * (1 - lambda_2), (1 - self.omega1) * lambda_2 + (1 - self.omega2) * (1 - lambda_2)]) + self.epsilon_o/2) / (1. + self.epsilon_o)

            f_val = f_val - 2*alpha * R * np.log(np.sqrt(p_1).dot(np.sqrt(p_2)))

            return f_val

        def gradf(lambda_ds):
            lambda_1 = lambda_ds[0]
            lambda_2 = lambda_ds[1]

            # start with the terms that don't depend on POVMs
            # gradient with respect to lambda_1
            gradf_lambda_1_val = -1
            # gradient with respect to lambda_2
            gradf_lambda_2_val = 1

            # number of repetitions of the POVM measurement
            R = self.R

            # the probability distributions corresponding to the POVM:
            # p_1^{i}(k) = (<E^{i}_k, sigma_1> + \epsilon_o/Nm)/(1 + \epsilon_o) and
            # p_2^{i}(k) = (<E^{i}_k, sigma_2> + \epsilon_o/Nm)/(1 + \epsilon_o)
            p_1 = (np.array([self.omega1 * lambda_1 + self.omega2 * (1 - lambda_1), (1 - self.omega1) * lambda_1 + (1 - self.omega2) * (1 - lambda_1)]) + self.epsilon_o/2) / (1. + self.epsilon_o)
            p_2 = (np.array([self.omega1 * lambda_2 + self.omega2 * (1 - lambda_2), (1 - self.omega1) * lambda_2 + (1 - self.omega2) * (1 - lambda_2)]) + self.epsilon_o/2) / (1. + self.epsilon_o)

            # Hellinger affinity between p_1 and p_2
            AffH = np.sqrt(p_1).dot(np.sqrt(p_2))

            # gradient with respect to lambda_1
            gradf_lambda_1_val = gradf_lambda_1_val - alpha * R * (self.omega1 - self.omega2) * np.sqrt(p_2/p_1).dot(np.array([1, -1]))/(AffH * (1. + self.epsilon_o))

            # gradient with respect to lambda_2
            gradf_lambda_2_val = gradf_lambda_2_val - alpha * R * (self.omega1 - self.omega2) * np.sqrt(p_1/p_2).dot(np.array([1, -1]))/(AffH * (1. + self.epsilon_o))

            # gradient with respect to lambda_ds
            gradf_val = np.array([gradf_lambda_1_val, gradf_lambda_2_val])

            return gradf_val

        # the other part of the objective function is an indicator function on X x X, so it is set to zero because all iterates in Nesterov's
        # second method are inside the domain
        P = lambda lambda_ds: 0.

        # proximal operator of an indicator function is a projection
        def prox_lP(lambda_ds, l, tol):
            # we project each component of lambda_ds into the unit interval [0, 1]

            lambda_1_projection = project_on_box(lambda_ds[0], 0, 1)

            lambda_2_projection = project_on_box(lambda_ds[1], 0, 1)

            lambda_ds_projection = np.array([lambda_1_projection, lambda_2_projection])

            return lambda_ds_projection

        # perform the minimization using Nesterov's second method (accelerated proximal gradient)
        lambda_ds_opt, error = minimize_proximal_gradient_nesterov(f, P, gradf, prox_lP, self.mpdm_lambda_ds_o, tol = self.tol, return_error = True)

        # check if tolerance is satisfied
        if error > self.tol:
            self.success = False
            warnings.warn("The tolerance for the optimization was not achieved. The estimates may be unreliable. Consider using a random initial condition by setting random_init = True.", MinimaxOptimizationWarning)

        # store the optimal point as initial condition for future use
        self.mpdm_lambda_ds_o = lambda_ds_opt

        # obtain the density matrices at the optimum
        self.lambda_1_opt = lambda_ds_opt[0]
        self.lambda_2_opt = lambda_ds_opt[1]

        return (self.lambda_1_opt, self.lambda_2_opt, -f(lambda_ds_opt))

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
        # print progress, if required
        if self.print_progress:
            print("Beginning optimization".ljust(22), end = "\r", flush = True)

        def Phi_r_bar_alpha(alpha):
            Phi_r_bar_alpha_val = alpha*self.r + 0.5*self.maximize_Phi_r_alpha_density_matrices(alpha = alpha)[2]

            return Phi_r_bar_alpha_val

        # perform the minimization
        alpha_optimization_result = sp.optimize.minimize_scalar(Phi_r_bar_alpha, bounds = (1e-16, 1e3), method = 'bounded')

        # value of alpha at optimum
        self.alpha_opt = alpha_optimization_result.x

        # value of objective function at optimum: gives the risk
        self.Phi_r_bar_alpha_opt = alpha_optimization_result.fun

        # print progress, if required
        if self.print_progress:
            print("Optimization complete".ljust(22))

        # check if alpha optimization was successful
        if not alpha_optimization_result.success:
            self.success = False
            warnings.warn("The optimization has not converge properly to the saddle-point. The estimates may be unreliable. Consider using a random initial condition by setting random_init = True.", MinimaxOptimizationWarning)

        return (self.lambda_1_opt, self.lambda_2_opt, self.alpha_opt)
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

            The above is the general procedure to obtain Juditsky & Nemirovski's estimator.
            For the special case of randomized Pauli measurement strategy, we simplify the above algorithms
            so that we can compure the estimator for very large dimensions.
        """
        # find x, y, and alpha components of the saddle point
        lambda_1_opt, lambda_2_opt, alpha_opt = self.find_density_matrices_alpha_saddle_point()

        # the saddle point value of \Phi_r
        Phi_r_opt = self.Phi_r_bar_alpha_opt

        # construct (phi/alpha)* at saddle point using lambda_1* and lambda_2*
        # the probability distributions corresponding to sigma_1*, sigma_2*:
        # p^{i}_1(k) = (<E^{i}_k, sigma_1*> + \epsilon_o/Ni)/(1 + \epsilon_o) and
        # p^{i}_2(k) = (<E^{i}_k, sigma_2*> + \epsilon_o/Ni)/(1 + \epsilon_o)
        p_1_opt = (np.array([self.omega1 * lambda_1_opt + self.omega2 * (1 - lambda_1_opt), (1 - self.omega1) * lambda_1_opt + (1 - self.omega2) * (1 - lambda_1_opt)]) + self.epsilon_o/2) / (1. + self.epsilon_o)
        p_2_opt = (np.array([self.omega1 * lambda_2_opt + self.omega2 * (1 - lambda_2_opt), (1 - self.omega1) * lambda_2_opt + (1 - self.omega2) * (1 - lambda_2_opt)]) + self.epsilon_o/2) / (1. + self.epsilon_o)

        # (phi/alpha)* at the saddle point
        phi_alpha_opt = 0.5*np.log(p_1_opt/p_2_opt)

        # obtain phi* at the saddle point
        self.phi_opt = phi_alpha_opt * self.alpha_opt

        # find the constant in the estimator
        # c = 0.5 (Tr(rho sigma_1*) + Tr(rho sigma_2*)) = 0.5 (lambda_1* + lambda_2*)
        self.c = 0.5*(lambda_1_opt + lambda_2_opt)

        # build the estimator
        def estimator(data):
            """
                Given R independent and identically distributed elements from \Omega = {1, 2} (2 possible outcomes) sampled
                as per p_{A(sigma)}, gives the estimate for the fidelity F(rho, sigma) = Tr(rho sigma).

                \hat{F}(\omega_1, ..., \omega_R)
                                    = \sum_{l = 1}^{R_i} phi^{i}*(\omega^{i}_l) + c
            """
            # if a list of list is provided, following convention for Fidelity_Estimation_Manager, we obtain the list of data inside
            if type(data[0]) in [list, tuple, np.ndarray]:
                data = data[0]

            # ensure that only data has only R elements (i.e., R repetitions), because the estimator is built for just that case
            if len(data) != self.R:
                raise ValueError("The estimator is built to handle only %d outcomes, while %d outcomes have been supplied." %(self.R, len(data)))

            # start with the terms that don't depend on the POVMs
            estimate = self.c

            # build the estimate using the phi* component at the saddle point, accounting for data from the POVM
            estimate = estimate + np.sum([self.phi_opt[l] for l in data])

            return estimate

        self.estimator = estimator

        return (estimator, Phi_r_opt)
    ###----- Constructing the fidelity estimator

def generate_sampled_pauli_measurement_outcomes(rho, sigma, R, num_povm_list, epsilon_o, flip_outcomes = False):
    """
        Generates the outcomes (index pointing to appropriate POVM element) for a Pauli sampling measurement strategy.
        The strategy involves sampling the non-identity Pauli group elements, measuring them, and only using the
        eigenvalue (either +1 or -1) of the measured outcome.
        The sampling is done as per the probability distribution p_i = |tr(W_i rho)| / \sum_i |tr(W_i rho)|.
        We represent this procedure by an effective POVM containing two elements.
        If outcome eigenvalue is +1, that corresponds to index 0 of the effective POVM, while eigenvalue -1 corresponds to index 1 of the effective POVM.

        If flip_outcomes is True, we measure the measure Paulis, and later flip the measurement outcomes (+1 <-> -1) as necessary. If not, we directly
        measure negative of the Pauli operator.

        The function requires the target state (rho) and the actual state "prepared in the lab" (sigma) as inputs.
        The states (density matrices) are expected to be flattened in row-major style.
    """
    # dimension of the system; rho is expected to be flattened, but this expression is agnostic to that
    n = int(np.sqrt(rho.size))

    # number of qubits
    nq = int(np.log2(n))
    if 2**nq != n:
        raise ValueError("Pauli measurements possible only in systems of qubits, i.e., the dimension should be a power of 2")

    # ensure that the states are flattened
    rho   = rho.ravel()
    sigma = sigma.ravel()

    # index of each Pauli of which weights need to be computed
    pauli_index_list = range(1, 4**nq)
    # find Tr(rho W) for each Pauli operator W (identity excluded); this is only a heuristic weight if rho is not pure
    # these are not the same as Flammia & Liu weights
    # computing each Pauli operator individulally (as opposed to computing a list of all Pauli operators at once) is a little slower, but can handle more number of qubits
    pauli_weight_list = [np.real(np.conj(rho).dot(generate_Pauli_operator(nq = nq, index_list = pauli_index, flatten = True)[0])) for pauli_index in pauli_index_list]

    # phase of each pauli operator (either +1 or -1)
    pauli_phase_list = [np.sign(pauli_weight) for pauli_weight in pauli_weight_list]

    # set of pauli operators along with their phases from which we will sample
    pauli_measurements = list(zip(pauli_index_list, pauli_phase_list))

    # probability distribution for with which the Paulis should be sampled
    pauli_sample_prob = np.abs(pauli_weight_list)
    # normalization factor for pauli probability
    NF = np.sum(pauli_sample_prob)
    # normalize the sampling probability
    pauli_sample_prob = pauli_sample_prob / NF

    # the effective POVM for minimax optimal strategy consists of just two POVM elements
    # however, the actual measurements performed are 'R' Pauli measurements which are uniformly sampled from the pauli operators
    # np.random.choice doesn't allow list of tuples directly, so indices are sampled instead
    # see https://stackoverflow.com/questions/30821071/how-to-use-numpy-random-choice-in-a-list-of-tuples/55517163
    uniformly_sampled_indices = np.random.choice(len(pauli_measurements), size = int(R), p = pauli_sample_prob)
    pauli_to_measure_with_repetitions = [pauli_measurements[index] for index in uniformly_sampled_indices]
    # unique Pauli measurements to be performed, with phase
    pauli_to_measure = sorted(list(set(pauli_to_measure_with_repetitions)), key = lambda x: x[0])

    # get the number of repetitions to be performed for each unique Pauli measurement (i.e., number of duplicates)
    R_list, _ = np.histogram([pauli_index for (pauli_index, _) in pauli_to_measure_with_repetitions], bins = [pauli_index for (pauli_index, _) in pauli_to_measure] + [pauli_to_measure[-1][0] + 1], density = False)

    # list of number of POVM elements for each (type of) measurement
    # if a number is provided, a list (of integers) is created from it
    if type(num_povm_list) not in [list, tuple, np.ndarray]:
        num_povm_list = [int(num_povm_list)] * len(R_list)
    else:
        num_povm_list = [int(num_povm) for num_povm in num_povm_list]

    # generate POVMs for measurement
    POVM_list = [None] * len(R_list)
    for (count, num_povm) in enumerate(num_povm_list):
        # index of pauli opetator to measure, along with the phase
        pauli, phase = pauli_to_measure[count]

        if flip_outcomes:
            # don't include the phase while measuring
            # the phase is incorporated after the measurement outcomes are obtained
            phase = 1

        # generate POVM depending on whether projectors on subpace or projectors on each eigenvector is required 
        # note that when n = 2, subspace and eigenbasis projectors match, in which case we give precedence to eigenbasis projection
        # this is because in the next block after measurements are generated, we check if num_povm is n and if that's true include phase
        # but if subspace was used first, then phase would already be included and this would be the same operation twice
        # so we use check for eigenbasis projection first
        if num_povm == n:
            # ensure that the supplied Pauli operator is a string composed of 0, 1, 2, 3
            if type(pauli) in [int, np.int64]:
                if pauli > 4**nq - 1:
                    raise ValueError("Each pauli must be a number between 0 and 4^{nq} - 1")
                # make sure pauli is a string
                pauli = np.base_repr(pauli, base = 4)
                # pad pauli with 0s on the left so that the total string is of size nq (as we need a Pauli operator acting on nq qubits)
                pauli = pauli.rjust(nq, '0')
            elif type(pauli) == str:
                # get the corresponding integer
                pauli_num = np.array(list(pauli), dtype = 'int')
                pauli_num = pauli_num.dot(4**np.arange(len(pauli) - 1, -1, -1))
                
                if pauli_num > 4**nq - 1:
                    raise ValueError("Each pauli must be a number between 0 and 4^{nq} - 1")

                # pad pauli with 0s on the left so that the total string is of size nq (as we need a Pauli operator acting on nq qubits)
                pauli = pauli.rjust(nq, '0')

            # we take POVM elements as rank 1 projectors on to the (orthonormal) eigenbasis of the Pauli operator specified by 'pauli' string
            # - first create the computation basis POVM and then use the Pauli operator strings to get the POVM in the respective Pauli basis
            computational_basis_POVM = generate_POVM(n = n, num_povm = n, projective = True, pauli = None, flatten = False, isComplex = True, verify = False)

            # - to get Pauli X basis, we can rotate the computational basis using Hadamard
            # - to get Pauli Y basis, we can rotate the computational basis using a matrix similar to Hadamard
            # use a dictionary to make these mappings
            comp_basis_transform_dict = {'0': np.eye(2, dtype = 'complex128'), '1': np.array([[1., 1.], [1., -1.]], dtype = 'complex128')/np.sqrt(2),\
                                         '2': np.array([[1., 1.], [1.j, -1.j]], dtype = 'complex128')/np.sqrt(2), '3': np.eye(2, dtype = 'complex128')}
            transform_matrix = np.eye(1)
            # pauli contains tensor product of nq 1-qubit Pauli operators, so parse through them to get a unitary mapping computational basis to Pauli eigenbasis
            for ithpauli in pauli:
                transform_matrix = np.kron(transform_matrix, comp_basis_transform_dict[ithpauli])

            # create the POVM by transforming the computational basis to given Pauli basis
            # the phase doesn't matter when projecting on to the eigenbasis; the eigenvalues are +1, -1 or +i, -i, depending on the phase but we can infer that upon measurement
            POVM = [transform_matrix.dot(Ei).dot(np.conj(transform_matrix.T)).ravel() for Ei in computational_basis_POVM]
        elif num_povm == 2:
            # the Pauli operator that needs to be measured
            Pauli_operator = phase * generate_Pauli_operator(nq, pauli)[0]

            # if W is the Pauli operator and P_+ and P_- are projectors on to the eigenspaces corresponding to +1 (+i) & -1 (-i) eigenvalues, then
            # l P_+ - l P_- = W, and P_+ + P_- = \id. We can solve for P_+ and P_- from this. l \in {1, i}, depending on the pase.
            # l = 1 or i can be obtained from the phase as sgn(phase) * phase, noting that phase is one of +1, -1, +i or -i
            P_plus  = 0.5*(np.eye(n, dtype = 'complex128') + Pauli_operator / (phase * np.sign(phase)))
            P_minus = 0.5*(np.eye(n, dtype = 'complex128') - Pauli_operator / (phase * np.sign(phase)))

            POVM = [P_plus.ravel(), P_minus.ravel()]
        else:
            raise ValueError("Pauli measurements with only 2 or 'n' POVM elements are supported")

        # store the POVM for measurement
        POVM_list[count] = POVM

    # initiate the measurements
    measurement_manager = Measurement_Manager(random_seed = None)
    measurement_manager.n = n
    measurement_manager.N = len(POVM_list)
    measurement_manager.POVM_mat_list = [np.vstack(POVM) for POVM in POVM_list]
    measurement_manager.N_list = [len(POVM) for POVM in POVM_list]

    # perform the measurements
    data_list = measurement_manager.perform_measurements(sigma, R_list, epsilon_o, num_sets_outcomes = 1, return_outcomes = True)[0]

    # convert the outcomes of the Pauli measurements to those of the effective POVM
    effective_outcomes = list()
    for (count, data) in enumerate(data_list):
        num_povm = num_povm_list[count]
        pauli_index, phase = pauli_to_measure[count]

        if flip_outcomes:
            # store the actual phase for later use
            actual_phase = int(phase)
            # Pauli were measured without the phase, so do the conversion of outcomes to those of effective POVM with that in mind
            phase = 1
        
        # for num_povm = 2, there is nothing to do because outcome '0' corresponds to +1 eigenvalue and outcome 1 corresponds to -1 eigenvalue
        # if flip_outcomes is False, then these are also the outcomes for the effective POVM because phase was already accounted for during measurement
        # if flip_outcomes is True, then we will later flip the outcome index (0 <-> 1) to account for the phase

        # for num_povm = n, we need to figure out the eigenvalue corresponding to outcome (an index from 0 to n - 1, pointing to the basis element)
        # we map +1 value to 0 and -1 eigenvalue to 1, which corresponds to the respective indices of elements in the effective POVM
        if num_povm == n:
            # all Paulis have eigenvalues 1, -1, but we are doing projective measurements onto the eigenbasis of Pauli operators
            # so, half of them will have +1 eigenvalue, the other half will have -1 eigenvalue
            # we are mapping the computational basis to the eigenbasis of the Pauli operator to perform the measurement
            # 0 for the ith qubit goes to the +1 eigenvalue eigenstate of the ith Pauli, and
            # 1 for the ith qubit goes to the -1 eigenvalue eigenstate of the ith Pauli
            # the exception is when the ith Pauli is identity, where the eigenstate is as described above but eigenvalue is always +1
            # therefore, we assign an "eigenvalue weight" of 1 to non-identity 1-qubit Paulis (X, Y, Z) and an "eigenvalue weight" of 0 to the 1-qubit identity
            # we then write the nq-qubit Pauli string W as an array of above weights w_1w_2...w_nq, where w_i is the "eigenvalue weight" of the ith Pauli in W
            # then the computational basis state |i_1i_2...i_nq> has the eigenvalue (-1)^(i_1*w_1 + ... + i_nq*w_nq) when it has been transformed to an
            # however, if the Pauli operator has a non-identity phase, the +1 and -1 eigenvalue are appropriately changed
            # the general expression for eigenvalue takes the form phase * (-1)^(i_1*w_1 + ... + i_nq*w_nq)
            # eigenstate of the Pauli operator W (using the transform_matrix defined in qi_utilities.generate_POVM)
            # so given a pauli index (a number from 0 to 4^nq - 1), obtain the array of "eigenvalue weight" representing the Pauli operator as described above
            # for this, convert the pauli index to an array of 0, 1, 2, 3 representing the Pauli operator (using np.base_repr, np.array), then set non-zero elements to 1 (using np.where)
            pauli_eigval_weight = lambda pauli_index: np.where(np.array(list(np.base_repr(pauli_index, base = 4).rjust(nq, '0')), dtype = 'int8') == 0, 0, 1)
            # get array of 0, 1 representing the computational basis element from the index (a number from 0 to 2^nq - 1) of the computational basis
            computational_basis_array = lambda computational_basis_index: np.array(list(np.base_repr(computational_basis_index, base = 2).rjust(nq, '0')), dtype = 'int8')
            # for the eigenvalues from the (computational basis) index of the outcome for each pauli measurement performed
            # to convert the eigenvalue (+1 or -1) to index (0 or 1, respectively), we do the operation (1 - e) / 2, where e is the eigenvalue
            # type-casted to integers because an index is expected as for each outcome
            data = [int(np.real( (1 - phase*(-1)**(computational_basis_array(outcome_index).dot(pauli_eigval_weight(pauli_index)))) / 2 )) for outcome_index in data]

        if flip_outcomes and actual_phase == -1:
            # now that we have the data for the effective POVM (without considering the phase), we can flip the outcomes as necessary
            data = [1 - outcome_index for outcome_index in data]

        # include this in the list of outcomes for the effective measurement
        effective_outcomes.extend(data)

    return effective_outcomes

def fidelity_estimation_pauli_random_sampling(target_state = 'random', nq = 2, num_povm_list = 2, R = 100, epsilon = 0.05, risk = None, epsilon_o = 1e-5, noise = True,\
                                              noise_type = 'depolarizing', state_args = None, flip_outcomes = False, tol = 1e-6, random_seed = 1, verify_estimator = False,\
                                              print_result = True, write_to_file = False, dirpath = './Data/Computational/', filename = 'temp'):
    """
        Generates the target_state defined by 'target_state' and state_args, and finds an estimator for fidelity using Juditsky & Nemirovski's approach for a specific measurement scheme
        involving random sampling of Pauli operators.
        The specialized approach allows for computation of the estimator for very large dimensions.

        The random sampling is done as per the probability distribution p_i = |tr(W_i rho)| / \sum_i |tr(W_i rho)|, where W_i is the ith Pauli operator and rho is the target state.
        This random sampling is accounted for by a single POVM, so number of types of measurement (N) is just one.
        The estimator and the risk only depend on the dimension, the number of repetitions, the confidence level, and the normalization factor NF = \sum_i |tr(W_i rho)|.

        If risk is a number less than 0.5, the number of repetitions of the minimax optimal measurement is chosen so that the risk of the estimator is less than or equal to the given risk.
        The argument R is ignored in this case.

        Checks are not performed to ensure that the given set of generators indeed form generators.

        If verify_estimator is true, the estimator constructed for the special case of randomized Pauli measurement strategy is checked with the general construction
        for Juditsky & Nemirovski's estimator.
    """
    # set the random seed once here and nowhere else
    if random_seed:
        np.random.seed(int(random_seed))

    # number of qubits
    nq = int(nq)
    # dimension of the system
    n  = int(2**nq)

    ### create the states
    # create the target state from the specified generators
    target_state = str(target_state).lower()
    if target_state in ['ghz', 'w', 'cluster']:
        state_args_dict = {'ghz': {'d': 2, 'M': nq}, 'w': {'nq': nq}, 'cluster': {'nq': nq}}

        rho = generate_special_state(state = target_state, state_args = state_args_dict[target_state], density_matrix = True,\
                                     flatten = True, isComplex = True)
    elif target_state == 'stabilizer':
        generators = state_args['generators']

        # if generators are specified using I, X, Y, Z, convert them to 0, 1, 2, 3
        generators = [g.lower().translate(str.maketrans('ixyz', '0123')) for g in generators]

        rho = generate_special_state(state = 'stabilizer', state_args = {'nq': nq, 'generators': generators}, density_matrix = True, flatten = True, isComplex = True)
    elif target_state == 'random':
        rho = generate_random_state(n = n, pure = True, density_matrix = True, flatten = True, isComplex = True, verify = False, random_seed = None)
    else:
        raise ValueError("Please specify a valid target state. Currently supported arguments are GHZ, W, Cluster, stabilizer and random.")

    # apply noise to the target state to create the actual state ("prepared in the lab")
    if not ((noise is None) or (noise is False)):
        # the target state decoheres due to noise
        if type(noise) in [int, float]:
            if not (noise >= 0 and noise <= 1):
                raise ValueError("noise level must be between 0 and 1")

            sigma = depolarizing_channel(rho, p = noise)
        else:
            sigma = depolarizing_channel(rho, p = 0.1)
    else:
        sigma = generate_random_state(n, pure = False, density_matrix = True, flatten = True, isComplex = True, verify = False,\
                                      random_seed = None)

    ### generate the measurement outcomes for the effective (minimax optimal) POVM
    # calculate the normalization factor
    # computing each Pauli operator individulally (as opposed to computing a list of all Pauli operators at once) is a little slower, but can handle more number of qubits
    NF = np.sum([np.abs(np.conj(rho).dot(generate_Pauli_operator(nq = nq, index_list = pauli_index, flatten = True)[0])) for pauli_index in range(1, 4**nq)])

    # if risk is given, then choose the number of repetitions to achieve that risk (or a slightly lower risk)
    if risk is not None:
        if risk < 0.5:
            R = int(np.ceil(2*np.log(2/epsilon) / np.abs(np.log(1 - (n/NF)**2 * risk**2))))
        else:
            raise ValueError("Only risk < 0.5 can be achieved by choosing appropriate number of repetitions of the minimax optimal measurement.")
        
    effective_outcomes = generate_sampled_pauli_measurement_outcomes(rho, sigma, R, num_povm_list, epsilon_o, flip_outcomes)

    ### obtain the fidelity estimator
    PSFEM = Pauli_Sampler_Fidelity_Estimation_Manager(n, R, NF, epsilon, epsilon_o, tol)
    fidelity_estimator, risk = PSFEM.find_fidelity_estimator()

    # obtain the estimate
    estimate = fidelity_estimator(effective_outcomes)

    # verify the estimator created for the specialized case using the general approach
    if verify_estimator:
        # the effective POVM for the optimal measurement strategy is simply {omega_1 rho + omega_2 Delta_rho, (1 - omega_1) rho + (1 - omega_2) Delta_rho},
        # where omega_1 = (n + NF - 1)/2NF, omega_2 = (NF - 1)/2NF, and Delta_rho = I - rho
        omega1 = 0.5 * (n + NF - 1) / NF
        omega2 = 0.5 * (1 - 1/NF)
        Delta_rho = np.eye(2**nq).ravel() - rho
        POVM_list = [[omega1 * rho + omega2 * Delta_rho, (1 - omega1) * rho + (1 - omega2) * Delta_rho]]

        # Juditsky & Nemirovski estimator
        FEMC = Fidelity_Estimation_Manager_Corrected(R, epsilon, rho, POVM_list, epsilon_o, tol)
        fidelity_estimator_general, risk_general = FEMC.find_fidelity_estimator()

        # matrices at optimum
        sigma_1_opt, sigma_2_opt = embed_hermitian_matrix_real_vector_space(FEMC.sigma_1_opt, reverse = True, flatten = True), embed_hermitian_matrix_real_vector_space(FEMC.sigma_2_opt, reverse = True, flatten = True)
        # constraint at optimum
        constraint_general = np.real(np.sum([np.sqrt((np.conj(Ei).dot(sigma_1_opt) + epsilon_o/2)*(np.conj(Ei).dot(sigma_2_opt) + epsilon_o/2)) / (1 + epsilon_o) for Ei in POVM_list[0]]))

    if print_result:
        print("True fidelity", np.real(np.conj(rho).dot(sigma)))
        print("Estimate", estimate)
        print("Risk", risk)
        print("Repetitions", R)
        # print results from the general approach
        if verify_estimator:
            print("Risk (general)", risk_general)
            print("Constraint (general)", constraint_general, "Lower constraint bound", (epsilon / 2)**(1/R))

    if not verify_estimator:
        return PSFEM
    else:
        return (PSFEM, FEMC)
