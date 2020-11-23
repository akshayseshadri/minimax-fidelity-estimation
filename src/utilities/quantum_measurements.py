"""
    Manages creation of POVMs and geneartion of outcomes from measuring these POVM given a state

    Author: Akshay Seshadri
"""

import numpy as np
import scipy as sp
from scipy import stats

import project_root # noqa
from src.utilities.qi_utilities import generate_random_state, generate_POVM, generate_Pauli_operator
from src.utilities.noise_process import depolarizing_channel

class Measurement_Manager():
    """
        Manages creation of different types of POVMs, and generating measurement outcomes from these POVMs (for which a state is created)
    """
    def __init__(self, random_seed = 1):
        """
            Initializes some of the parameters
        """
        # store the random seed for access by other functions (mainly for including it in parameters to be saved)
        self.random_seed = random_seed
        # set the random seed only once when the class is invoked, and nowhere else
        if random_seed:
            np.random.seed(int(random_seed))

    def create_measurements(self, n = 2, N = 2, num_povm_list = 2, pauli = True, rho = None, projective_list = False, return_POVM = True):
        """
            Generates POVM using the specified parameters. The actual POVM generation is done by qi_utilities.generate_POVM; create_measurements
            generates appropriate parameters to pass to this function and does the necessary post-processing.

            'pauli' can either be True, False/None, or a list of pauli operators (represented by a string of the form 'i_1i_2...i_nq', where each i_k is between 0 & 3)
            If pauli is True, then a state "rho" must be provided; Pauli weights are obtained using this state
        """
        ### store the parameters for later use
        # dimension of the system
        self.n = n

        # number of (types of) measurements
        self.N = N

        # list of number of POVM elements for each (type of) measurement
        # if a number is provided, a list (of integers) is created from it
        if type(num_povm_list) != list:
            num_povm_list = [int(num_povm_list)]*N
        else:
            num_povm_list = [int(num_povm) for num_povm in num_povm_list]

        # list specifying whether to use projective measurements
        if type(projective_list) != list:
            projective_list = [projective_list]*N
            self.projective_list = projective_list
        else:
            projective_list = [bool(is_projective) for is_projective in projective_list]
            self.projective_list = projective_list

        # whether to use Pauli measurements
        self.pauli = pauli

        # generate the POVM
        # list of POVMs
        POVM_list = [0]*N

        # generate a POVM
        if not (pauli is None or pauli is False):
            # ensure that the systems is comprised of qubits
            # number of qubits
            nq = int(np.log2(n))
            if 2**nq != n:
                raise ValueError("Pauli weighting possible only in systems of qubits, i.e., the dimension should be a power of 2")

            # pauli can either be True, False/None, or a list of pauli operators (represented by a string of the form 'i_1i_2...i_nq', where each i_k is between 0 & 3)
            # if pauli is not a list, do weighted Pauli measurements
            if type(pauli) not in [tuple, list, np.ndarray]:
                if rho is None:
                    raise ValueError("A state must be provided to find the Pauli weights")

                # find Tr(rho W) for each Pauli operator W; this is only a heuristic weight if rho is not pure
                self.pauli_weight_list = [(count + 1, (np.abs(np.conj(rho).dot(W))/np.sqrt(n))**2) for (count, W) in\
                                                        enumerate(generate_Pauli_operator(nq, list(range(1, 4**nq)), flatten = True))]

                # find the largest 'N' weights, and measure the corresponding Pauli operators
                self.pauli_measurements = sorted(self.pauli_weight_list, key = lambda x: x[1], reverse = True)[:N]

                # Note: if the entries of num_povm_list are different, it is ambiguous which entry corresponds to the Pauli operator after the above re-ordering,
                #       so we don't allow for this; one can specify the pauli operators as well as num_povm_list explicitly (as a list) for using this option
                # see https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
                if [num_povm_list[0]] * len(num_povm_list) != num_povm_list:
                    raise ValueError("When Pauli operators are to be inferred using weights, every element of num_povm_list must be the same")
                else:
                    num_povm = num_povm_list[0]

                # build a POVM for each selected Pauli measurement
                POVM_list = [generate_POVM(n, num_povm = num_povm, projective = True, flatten = True, isComplex = True, verify = False,\
                                           pauli = np.base_repr(i, base = 4), random_seed = None) for (i, _) in self.pauli_measurements]
            else:
                if len(pauli) != N:
                    raise ValueError("Number of measurements (N) provided and the number of pauli operators listed do not match")
                # use the strings specified in pauli to measure those Pauli operators
                POVM_list = [generate_POVM(n, num_povm = num_povm, projective = True, flatten = True, isComplex = True, verify = False,\
                                           pauli = pauli_string, random_seed = None) for (num_povm, pauli_string) in zip(num_povm_list, pauli)]
        else:
            for i in range(N):
                num_povm = num_povm_list[i]

                # when projective is True, ensure that num_povm = n
                if projective_list[i]:
                    num_povm = n

                POVM_list[i] = generate_POVM(n, num_povm = num_povm, projective = projective_list[i], flatten = True, isComplex = True, verify = False,\
                                             random_seed = None)

        # list of POVM elements corresponding to each measurement
        self.POVM_list = POVM_list

        # list of POVM elements flattened and stacked to be stored as one large matrix for each measurement
        self.POVM_mat_list = [np.vstack(POVM) for POVM in POVM_list]

        # we also store N_list (which is basically num_povm_list), but this naming convention is used for other parts of the code
        self.N_list = [len(POVM) for POVM in POVM_list]

        if return_POVM:
            return POVM_list

    def perform_measurements(self, sigma = None, R_list = 1000, epsilon_o = 1e-5, outcome_eigvals = None, num_sets_outcomes = 1, pure = True, noise = False,\
                             noise_type = 'depolarizing', return_outcomes = True):
        """
            Performs measurements specified by generate_POVM. The number of repetitions of each type of measurement is specified by R_list.

            For performing the measurement, a state (of the system) 'sigma' can be specified.
            If this is not specified, a state is generated. This can either be pure or mixed (speified by 'pure').
            The state can optionally be passed through a noise channel. The type of noise channel is specified by 'noise_type', and
            whether or how much noise to add is specified by 'noise'.

            sigma             : state of the system
            R_list            : a list of number of repetitions of each type of measurement
            epsilon_o         : constant incorporated in Born's rule to avoid zero division (a positive number much smaller than 1)
            outcome_eigvals   : The eigenvalues of the outcomes described by the POVM (each eigenvalue should correspond to the
                                appropriate POVM element).
                                Default is None, which means instead of the eigenvalue, the index of the POVM is returned.
                                Should have the format [[Ni elts] for Ni in range(N_list)]. Checks are not performed to ensure this is the case.
            num_sets_outcomes : one set of outcomes is each of the measurements repeated a number of times specified by R_list;
                                'num_sets_outcomes' specifies how many such sets are generated
            pure              : specifies whether the state of the system is pure or mixed
            noise             : if True/False, specifies whether to pass the generated state through a noise channel
                                if True, a noise value of 0.1 is used
                                if noise is a number between 0 and 1, that noise level is used
            noise_type        : specifies which noise channel to use

            TODO: If num_sets_outcomes = 1, return the outcomes directly, instead of nested them in a list.
                  Not changing this behaviour now as it would break some other code.
        """
        # constant in modified Born's rule that prevents zero-division
        self.epsilon_o = epsilon_o

        # list of number of repetitions of each (type of) measurement
        if type(R_list) not in [tuple, list, np.ndarray]:
            self.R_list = [int(R_list)]*self.N
        else:
            # convert R to an integer if it already isn't one
            self.R_list = [int(R) for R in R_list]

        num_sets_outcomes = int(num_sets_outcomes)

        ### parameters related to the state
        # whether to create a pure or mixed state
        self.pure = pure

        # whether to pass the prepared state through a noisy channel
        # note that noise can be True, False or any number between 0 and 1
        self.noise = noise

        # type of noise channel
        self.noise_type = noise_type

        # generate the state
        # create the state ("prepared in the lab")
        if sigma is None:
            sigma = generate_random_state(self.n, pure = pure, density_matrix = True, flatten = True, isComplex = True, verify = False,\
                                          random_seed = None)

            if noise:
                # the state decoheres due to noise
                if type(noise) == float:
                    if not (noise >= 0 and noise <= 1):
                        raise ValueError("noise level must be between 0 and 1")

                    sigma = depolarizing_channel(sigma, p = noise)
                else:
                    sigma = depolarizing_channel(sigma, p = 0.1)

        ### perform the measurements
        # check if we have POVM_list to perform the measurement
        try:
            POVM_mat_list = self.POVM_mat_list
        except AttributeError:
            raise ValueError("Measurements to be performed have not been specified. Use Measurement_Manager.create_measurements to generate the necessary measurements.")

        # assign eigenvalues the POVM outcomes if provided, else return the index of the POVM 
        if outcome_eigvals is None or outcome_eigvals is False:
            # point to the index of the outcome
            self.outcome_indicator = [list(range(Ni)) for Ni in self.N_list]
        else:
            # if eigenvalues are provided, generate them
            self.outcome_indicator = outcome_eigvals

        # a (discrete) random variable present with probabilities from Born's rule for each measurement
        self.drv_list = [None]*self.N
        self.p_sigma_list = [np.zeros(Ni) for Ni in self.N_list]

        # first obtain the probability distribution corresponding to each POVM measurement, and create a random variable with these
        # probability distributions for repeated use
        for i in range(self.N):
            # matrix for ith POVM
            POVM_mat_i = POVM_mat_list[i]
            # number of repetitions for ith measurement
            Ri = self.R_list[i]
            # number of possible outcomes for ith measurement
            Ni = self.N_list[i]
            # outcomes to be generated for the ith measurement
            outcome_indicator_i = self.outcome_indicator[i]

            # the probability distribution corresponding to the ith POVM in the actual state sigma
            p_sigma_i = np.real(np.conj(POVM_mat_i).dot(sigma) + epsilon_o/Ni) / (1. + epsilon_o)

            self.p_sigma_list[i] = p_sigma_i

            # create a list discrete random variable distributed as per p_sigma and taking values in outcome_indicator_i
            drv = sp.stats.rv_discrete(values = (outcome_indicator_i, p_sigma_i))

            self.drv_list[i] = drv

        # use the random variables to generate measurement outcomes
        self.num_sets_outcomes = num_sets_outcomes
        # Caution: don't do [[list]]*num in the following because this duplicates the inner list [list], and
        # changing the ith inner list will change all of the inner list
        self.data_list = [[np.zeros(Ri) for Ri in self.R_list] for _ in range(num_sets_outcomes)]
        for i_set in range(num_sets_outcomes):
            for i in range(self.N):
                # generate Ri samples from {0, ..., N - 1} as per the probability distribution p_sigma_i
                data_i = self.drv_list[i].rvs(size = self.R_list[i])

                # raw data is used to estimate fidelity directly
                self.data_list[i_set][i] = data_i

        if return_outcomes:
            return self.data_list
