"""
    Constructs a fidelity estimator from the specified target state and measurement settings.
    These can be provided using a YAML file.

    Some convenince functions are provided for Pauli measurements and special states.

    Author: Akshay Seshadri
"""

import numpy as np
from pathlib import Path
import yaml
import json
import csv
import warnings
from optparse import OptionParser

import project_root # noqa
from src.fidelity_estimation import Fidelity_Estimation_Manager
from src.fidelity_estimation_pauli_sampling import Pauli_Sampler_Fidelity_Estimation_Manager
from src.utilities.qi_utilities import generate_random_state, generate_special_state, generate_Pauli_operator, generate_POVM

### custom warning for optimization
class MinimaxOptimizationWarning(UserWarning):
    """
        Warning specific to optimization performed in the minimax method to find the saddle-point.
    """
    pass

### addtition to qi_utilities
def generate_Pauli_POVM(pauli, projection = 'eigenbasis', flatten = True, isComplex = True):
    """
        Given a Pauli operator as a string, constructs the POVM that performs the measurement of this Pauli operator.

        The string must consist of I, X, Y, Z or 0, 1, 2, 3. A phase of +1/-1, +i/-i can also be specified.
        The number of qubits are inferred from the Pauli operator.
        Note that the +i/-i phase must be input as +j/-j following Python convention.
        This also helps avoid confusion with the identity operator in case the string is in lower case.

        Two types of projective measurements are supported:
            'subspace'   - projection on +1, -1 eigenspaces of the Pauli operator
            'eigenbasis' - projection on each eigenvector of the Pauli operator

        Arguments:
            - pauli      : a string consisting of I, X, Y, Z describing the Pauli operator to be measured, along with a phase: -, j, or -j
            - projection : either subspace or eigenbasis (default: eigenbasis)
            - flatten    : whether to flatten (row major style) the elements of the POVM (default: True)
            - isComplex  : whether to typecast the POVM elements as complex numbers (default: True)
    """
    # the string specifying the Pauli operator
    pauli = str(pauli).lower()

    # get rid of + signs in the Pauli string
    pauli = pauli.strip('+')

    # extact the phase of the pauli operator
    if '-j' in pauli:
        phase = -1j
    elif '-' in pauli:
        phase = -1
    elif 'j' in pauli:
        phase = 1j
    else:
        phase = 1

    # remove the phase from the Pauli operator
    pauli = pauli.strip('-j')

    # convert I, X, Y, Z to 0, 1, 2, 3
    pauli = pauli.translate(str.maketrans('ixyz', '0123'))

    # number of qubits
    nq = len(pauli)

    # dimension of the system
    n = 2**nq

    # projection on subspace or eigenbasis
    projection = str(projection).lower()

    # generate POVM depending on whether projectors on subpace or projectors on each eigenvector is required
    if projection == 'subspace':
        # the Pauli operator that needs to be measured
        Pauli_operator = phase * generate_Pauli_operator(nq, pauli)[0]

        # if W is the Pauli operator (with phase included) and P_+ and P_- are projectors on to
        # the eigenspaces corresponding to +1 (+j) & -1 (-j) eigenvalues, then
        # l P_+ - l P_- = W, and P_+ + P_- = \id. We can solve for P_+ and P_- from this. l \in {1, j}, depending on the phase.
        # l = 1 or j can be obtained from the phase as sgn(phase) * phase, noting that phase is one of +1, -1, +j or -j
        P_plus  = 0.5*(np.eye(n, dtype = 'complex128') + Pauli_operator / (phase * np.sign(phase)))
        P_minus = 0.5*(np.eye(n, dtype = 'complex128') - Pauli_operator / (phase * np.sign(phase)))

        POVM = [P_plus.ravel(), P_minus.ravel()]
    elif projection == 'eigenbasis':
        # ensure that the supplied Pauli operator is a valid Pauli operator 
        # get the corresponding integer
        pauli_num = np.array(list(pauli), dtype = 'int')
        pauli_num = pauli_num.dot(4**np.arange(len(pauli) - 1, -1, -1))
        
        if pauli_num > 4**nq - 1:
            raise ValueError("Each pauli must be a number between 0 and 4^{nq} - 1")

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
        # the phase doesn't matter when projecting on to the eigenbasis; the eigenvalues are +1, -1 or +j, -j, depending on the phase, but we can infer that upon measurement
        POVM = [transform_matrix.dot(Ei).dot(np.conj(transform_matrix.T)).ravel() for Ei in computational_basis_POVM]
    else:
        raise ValueError("Only projection on 'subspace' or 'eigenbasis' is supported.")

    return POVM

### parsing the settings, constructing the estimator and estimating the fidelity from the outcomes
def parse_yaml_settings_file(yaml_filepath):
    """
        Parses the YAML file specifying the target state, POVM list, confidence level, and other settings.

        Constructs the target state and POVM list and returns them along with the other settings.

        Arguments:
            - yaml_filepath: The filepath to the YAML file containing the settings.
    """
    with open(yaml_filepath) as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    ### parse and generate the target state
    try:
        target = yaml_data['target']
    except KeyError:
        raise ValueError("'target' must be provided. Call show_yaml_file_instructions for details.")

    # the target state is explicitly given or a special state has beeen provided
    if type(target) == list:
        if len(target) == 0:
            raise ValueError("Empty list supplied as a target state.")

        # the target state has been explicitly provided as a list of numbers
        if not type(target[0]) == dict:
            # if the state has been specified using complex numbers, YAML typecasts the entries as strings
            # if these strings have spaces in them, it cannot be typecast directly into a complex number
            # so simply convert everything to strings without spaces, and then typecast them later
            rho = np.asarray(["".join(str(val).split()) for val in target])
        # the target state is a special state
        else:
            # just consider the dictionary supplied to the target state
            target = target[0]
            allowed_special_states = ['ghz', 'w', 'cluster', 'werner', 'stabilizer', 'random']
            # ensure that a valid state has been supplied
            special_state = [str(key) for key in target.keys()]
            # first check if more than one keys have been provided
            if len(special_state) > 1:
                raise ValueError("Only one target state can be specified. Call show_yaml_file_instructions for instructions." %special_state)
            # next, check that the provided key is a valid state
            special_state = special_state[0]
            if not special_state.lower() in allowed_special_states:
                raise ValueError("%s is not a valid target state. Call show_yaml_file_instructions for instructions." %special_state)

            if special_state.lower() == 'random':
                nq = int(target[special_state])
                rho = generate_random_state(n = 2**nq, pure = True, density_matrix = True, flatten = True, isComplex = True, verify = False, random_seed = None)
            else:
                if special_state.lower() == 'ghz':
                    state_args = {'d': 2, 'M': int(target[special_state])}
                elif special_state.lower() in ['w', 'cluster']:
                    state_args = {'nq': int(target[special_state])}
                elif special_state.lower() == 'werner':
                    state_args = {'nq': int(target[special_state][0]), 'p': float(target[special_state][1])}
                elif special_state.lower() == 'stabilizer':
                    generators = target[special_state]
                    # ensure that all generators specify the same number of qubits
                    generators_nq = [len(generator.lstrip('+-')) for generator in generators]
                    if not generators_nq == [generators_nq[0]]*len(generators_nq):
                        raise ValueError("All stabilizer generators must have the same number of Pauli operators.")
                    # number of qubits
                    nq = generators_nq[0]
                    state_args = {'nq': nq, 'generators': generators}

                # generate the special target state
                rho = generate_special_state(state = special_state, state_args = state_args, density_matrix = True, flatten = True, isComplex = True)
    # the target state is stored in a .npy file
    elif type(target) == str:
        with open(target) as target_datafile:
            rho = np.asarray(np.load(target_datafile))
    # any other setting is spurious
    else:
        raise ValueError("Please provide a valid setting for the target state. Call show_yaml_file_instructions for details.")

    # ensure that the target state is a flattened complex array
    try:
        rho = rho.ravel().astype('complex128')
    except ValueError:
        raise ValueError("Unable to parse the supplied value for 'target'. If a list of values has been provided, ensure that entries are valid numbers.")

    ### parse and generate POVM_list
    try:
        POVM_list = yaml_data['POVM_list']
    except KeyError:
        raise ValueError("'POVM_list' must be provided. Call show_yaml_file_instructions for details.")
    # POVM_list is either explicitly given or is a Pauli measurement or a combination of the two
    if type(POVM_list) == list:
        POVM_list_parsed = list()
        # checks whether Randomized Pauli measurements have been specified
        rpm_specified = False
        for POVM in POVM_list:
            # a POVM has been explicitly provided
            if type(POVM) == list:
                # if complex numbers have been provided with spaces in them, it cannot be typecast
                # so simply convert everything to strings without spaces, and then typecast them later
                POVM_parsed = list()
                for Ei in POVM:
                    Ei = np.asarray(["".join(str(val).split()) for val in Ei])
                    POVM_parsed.append(Ei)
                POVM_list_parsed.append(POVM_parsed)
            # Pauli measurements have been specified
            elif type(POVM) == dict:
                if 'pauli' in [key.lower() for key in POVM.keys()]:
                    pauli_list = POVM['pauli']
                    # ensure that we have a list
                    if type(pauli_list) != list:
                        pauli_list = [pauli_list]
                    pauli_list_lower = [str(option).lower() for option in pauli_list]

                    # the default measurement is projection on eigenbasis
                    projection = 'eigenbasis'
                    if 'subspace' in pauli_list_lower:
                        projection = 'subspace'

                    # find the format of Pauli measurements from pauli_list (three formats allowed)
                    if 'fls' in pauli_list_lower:
                        pauli_format = 'fls'
                    elif 'rpm' in pauli_list_lower:
                        pauli_format = 'rpm'
                    else:
                        pauli_format = 'general'

                    # dimension of the system
                    n = int(np.sqrt(rho.size))

                    # number of qubits
                    nq = int(np.log2(n))

                    if pauli_format == 'fls':
                        # use N Pauli operators chosen as per N largest Flammia & Liu's weights
                        try:
                            N = int(pauli_list[1])
                        except KeyError:
                            raise ValueError("The number of (non-identity) Pauli operators to be measured must be provided. Call show_yaml_file_instructions for details.")

                        if projection == 'subspace':
                            num_povm = 2
                        elif projection == 'eigenbasis':
                            num_povm = n

                        # find Tr(rho W) for each Pauli operator W; this is only a heuristic weight if rho is not pure
                        pauli_weight_list = [(count + 1, (np.abs(np.conj(rho).dot(W))/np.sqrt(n))**2) for (count, W) in\
                                                           enumerate(generate_Pauli_operator(nq, list(range(1, 4**nq)), flatten = True))]

                        # find the largest 'N' weights, and measure the corresponding Pauli operators
                        pauli_measurements = sorted(pauli_weight_list, key = lambda x: x[1], reverse = True)[:N]

                        POVM_list_generated = [generate_POVM(n = n, num_povm = num_povm, projective = True, flatten = True, isComplex = True,\
                                                             verify = False, pauli = pauli, random_seed = None) for (pauli, _) in pauli_measurements]
                    elif pauli_format == 'rpm':
                        # use the random Pauli measurement scheme
                        POVM_list_generated = [['rpm', n]]
                        rpm_specified = True
                    else:
                        # use the list of strings to create the Pauli measurements
                        # list of strings specifying Pauli operators
                        pauli_op_list = [pauli for pauli in pauli_list if pauli.lower() not in ['subspace', 'eigenbasis']]
                        # ensure that all the pauli strings provided act on the same number of qubits
                        # number of qubits specified by the pauli strings
                        pauli_list_nq = [len(pauli.lstrip('+-j')) for pauli in pauli_op_list]

                        if not (pauli_list_nq == [nq]*len(pauli_list_nq)):
                            raise ValueError("Every Pauli operator must act on the same number of qubits.")

                        POVM_list_generated = [generate_Pauli_POVM(pauli, projection, flatten = True, isComplex = True) for pauli in pauli_op_list]

                    POVM_list_parsed.extend(POVM_list_generated)

            if rpm_specified and len(POVM_list_parsed) > 1:
                raise ValueError("When Randomized Pauli measurement scheme is specified, other measurements are not allowed.")

    # a path to a '.npy' file containing the POVM list has been supplied
    elif type(POVM_list) == str:
        with open(POVM_list) as target_datafile:
            POVM_list_parsed = np.asarray(np.load(target_datafile))

    # any other setting is spurious
    else:
        raise ValueError("Please provide a valid setting for POVM_list. Call show_yaml_file_instructions for details.")

    # ensure that all POVM elements are flattened complex arrays
    if not rpm_specified:
        POVM_elt_size = list()
        for (i_povm, POVM) in enumerate(POVM_list_parsed):
            for (j_povm_elt, Ej) in enumerate(POVM):
                try:
                    POVM[j_povm_elt] = Ej.ravel().astype('complex128')
                    POVM_elt_size.append(POVM[j_povm_elt].size)
                except ValueError:
                    raise ValueError("Unable to parse the supplied value for 'POVM_list'. If a list of values has been provided, ensure that all the entries are valid numbers.")
            POVM_list_parsed[i_povm] = POVM

        # ensure that all the POVM element sizes are the same as the target state size
        if not (POVM_elt_size == [rho.size]*len(POVM_elt_size)):
            raise ValueError("The target state and all the POVM elements must have the same size")

    # number of repetitions for each measurement setting
    try:
        R_list = yaml_data['R_list']
    except KeyError:
        raise ValueError("'R_list' must be provided. Call show_yaml_file_instructions for details.")
    # convert R_list to a list of integers, one for each POVM
    if rpm_specified:
        if type(R_list) in [int, float]:
            R_list_parsed = int(R_list)
        elif type(R_list) == list:
            R_list_parsed = sum([int(R) for R in R_list])
        else:
            raise ValueError("Please provide a valid R_list. Call show_yaml_file_instructions for details.")
    else:
        if type(R_list) in [int, float]:
            R_list_parsed = [int(R_list)]*len(POVM_list_parsed)
        elif type(R_list) == list:
            # check that R_list and POVM_list_parsed have the same length
            if not len(R_list) == len(POVM_list_parsed):
                raise ValueError("The number of POVMs and number of repetitions (R) should be the same. Call show_yaml_file_instructions for details.")
            else:
                R_list_parsed = [int(R) for R in R_list]
        else:
            raise ValueError("Please provide a valid R_list. Call show_yaml_file_instructions for details.")

    # confidence level
    try:
        # epsilon = 1 - confidence_level is used in the algorithms
        epsilon = 1 - float(yaml_data['confidence_level'])
    except KeyError:
        raise ValueError("'confidence_level' must be provided. Call show_yaml_file_instructions for details.")
    if not (0 < epsilon < 0.25):
        raise ValueError("The confidence level should be between 0.75 and 1, with end points exluded.")

    # parse the optional arguments
    optional_args = dict()

    # random initial condition: optional argument
    try:
        random_init = yaml_data['random_init']
    except KeyError:
        # set the default random_init value
        random_init = False

    if not random_init in [True, False, 1, 0]:
        raise ValueError("'random_init' must either be True or False.")
    else:
        random_init = bool(random_init)
    optional_args['random_init'] = random_init

    return (rho, POVM_list_parsed, R_list_parsed, epsilon, optional_args)

def construct_fidelity_estimator(yaml_filename, estimator_filename, yaml_file_dir = './yaml_files', estimator_dir = './estimator_files', print_progress = True):
    """
        Constructs the Juditsky & Nemirovski estimator for fidelity for the specified target state and measurement settings.

        The specified target state and measurement settings must be provided using a YAML file.
        Details on how the YAML file must be constructed can be seen by calling the 'show_yaml_file_instructions' function.

        The estimator and the risk are saved in a JSON file. If a '.json' extension is not provided, it is appended to the filename.

        Arguments:
            - yaml_file          : Name of the YAML file containing details about the target state or measurement settings.
            - estimator_filename : Name of the file to which the constructed estimator should be saved.
            - yaml_file_dir      : Path to the directory where the YAML file is stored.
                                   (default: subdirectory of the current directory named 'yaml_files')
            - estimator_dir      : Path to the directory where the constructed estimator file should be saved.
                                   (default: subdirectory of the current directory named 'estimator_files'.
                                             If the subdirectory 'estimator_files' does not exist, it is created.)
            - print_progress     : Specifies whether the progress of optimization is printed.

        TODO: Test loading from .npy files specified in the YAML file settings.
    """
    ### ensure that the supplied YAML settings file exists, and the estimator data file doesn't already exist
    # get a proper path for the user's operating system
    # pathlib accepts forward slashes and internally changes it to whatever is appropriate for the OS
    yaml_filepath = Path('%s/%s' %(yaml_file_dir, yaml_filename))
    if not yaml_filepath.exists():
        raise ValueError("The specified YAML file does not exist. Please provide a valid file path.")

    # ensure that the estimator file doesn't already exist (to avoid over-writing)
    estimator_filepath = Path('%s/%s' %(estimator_dir, estimator_filename)).with_suffix('.json')
    if estimator_filepath.exists():
        raise ValueError("The specified estimator file already exists. Please provide a unique filename or delete that file.")

    ### get the settings required to construct the estimator
    # get the target state, measurement settings, and confidence level from the YAML file
    rho, POVM_list, R_list, epsilon, optional_args = parse_yaml_settings_file(yaml_filepath)

    # default values for tolerance & epsilon_o
    # tolerance to be used in optimization algorithms
    tol = 1e-6

    # the constant preventing the Born probabilities from going to zero, to avoid a zero value in logarithm
    epsilon_o = 1e-5

    # dimension of the system
    n = int(np.sqrt(rho.size))

    # sepcify whether to use random initial condition for performing the optimization
    random_init = optional_args['random_init']

    ### construct the fidelity estimator
    # if Randomized Pauli measurement scheme has been opted for, use a specially designed efficient algorithm
    # POVM_list[0][0] is equal to 'rpm' if Randomized Pauli measurement scheme has been specified
    if type(POVM_list[0][0]) == str:
        # the normalization factor is known for stabilizer states, so use a special algorithm for stabilizer states
        # check if the target state is a stabilizer state
        with open(yaml_filepath) as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
            target = yaml_data['target']
            del yaml_data

        # keep track of whether the state is a stabilizer state
        isStabilizer = False
        
        # since parsing has already been done, we don't need to perform sanity checks
        if type(target) == list:
            if type(target[0]) == dict:
                special_state = list(target[0].keys())[0]
                if special_state.lower() == 'stabilizer':
                    isStabilizer = True

        if isStabilizer:
            # the normalization factor is n - 1 for stabilizer states
            NF = n - 1
        else:
            # number of qubits in the system
            nq = int(np.log2(n))
            # compute the normalization factor required for constructing the fidelity estimator
            # computing each Pauli operator individulally (as opposed to computing a list of all Pauli operators at once) is a little slower, but can handle more number of qubits
            NF = np.sum([np.abs(np.conj(rho).dot(generate_Pauli_operator(nq = nq, index_list = pauli_index, flatten = True)[0])) for pauli_index in range(1, 4**nq)])

        # construct the fidelity estimator for Randomized Pauli measurement scheme
        PSFEM = Pauli_Sampler_Fidelity_Estimation_Manager(n, R_list, NF, epsilon, epsilon_o, tol, random_init, print_progress)
        _, risk = PSFEM.find_fidelity_estimator()
        phi_opt_list = [PSFEM.phi_opt]
        c = PSFEM.c
        success = PSFEM.success
    else:
        # construct the fidelity estimator for specified target state and measurement settings
        FEM = Fidelity_Estimation_Manager(R_list, epsilon, rho, POVM_list, epsilon_o, tol, random_init, print_progress)
        _, risk = FEM.find_fidelity_estimator()
        phi_opt_list = FEM.phi_opt_list
        c = FEM.c
        success = FEM.success

    ### save the estimator
    # create the necessary parent directories to save the estimator file if they don't already exist
    estimator_filepath.parent.mkdir(parents = True, exist_ok = True)

    # save R_list along with the estimator so as to check that the number of outcomes supplied are correct
    if type(R_list) not in [list, tuple, np.ndarray]:
        R_list = [R_list] * len(POVM_list)

    # save the estimator as a JSON file
    estimator_data = {'estimator': {'phi_opt_list': [phi_opt_i.tolist() for phi_opt_i in phi_opt_list], 'c': c}, 'risk': risk, 'R_list': R_list, 'success': success}

    with open(estimator_filepath, 'w') as datafile:
        json.dump(estimator_data, datafile)

def compute_fidelity_estimate_risk(outcomes, estimator_filename, estimator_dir = './estimator_files'):
    """
        Computes the fidelity estimate from the given measurement outcomes. The pre-computed risk is also returned.

        The outcomes must be in the same order as the POVMs specified to the 'construct_fidelity_estimator' function,
        while constructing the estimator.
        The outcomes can take the following form:
        1. A list of arrays (or lists), with each array corresponding to the outcomes for a particular POVM measurement.
        2. Path to a YAML file in the following format:
            outcomes:
                - outcomes for POVM 1
                - outcomes for POVM 2
                ...
        3. CSV files in one of the following formats:
            a. Path to a CSV file with each row corresponding to the outcomes for a particular POVM.
            b. A dictionary {'csv_file_path': Path to CSV file, 'entries': 'row', 'start': (row index, column index)}
                'entries' can take the value of 'row' or 'column'.
                    - If 'row', then each row must correspond to the outcomes for a particular POVM.
                    - If 'column', then each column must correspond to the outcomes for a particular POVM.
                'start' should be a tuple of non-negative integers denoting where the data starts.
                    - If 'start': (i, j), then the data starts at row 'i' and column 'j', in either row-wise or
                      column-wise format as specified by 'entries'.
                      Any entry in the CSV file before row 'i' and column 'j' is discarded.
                    Note that Python is zero-indexed, so the first row/column will be 0, second row/column will be 1, and so on.
               Example: 
                   If the data is stored in columns, but the top row is some description of the data, then we should specify:
                   {'csv_file_path': Path to CSV file, 'entries': 'column', 'start': (1, 0)}
        
           Note that Microsoft Excel files can be saved as CSV files.

        Important notes:
        1. If outcome_i denotes the array of outcomes for POVM_i = {E_1, ..., E_Ni}, then the entries of the outcomes_i
           must be 0, ..., Ni - 1, with 0 pointing to the POVM element E_1, 1 pointing to the POVM element E_2, and so on.
           We use a zero-index following Python convention.
        2. If Randomized Pauli measurement scheme was used to construct the estimator, the eigenvalues (+1, -1) must first
           be converted to indices (0, 1), before supplying it to 'compute_fidelity_estimate_risk' function.
           'convert_Pauli_eigvals_to_indices' function can be used for this purpose to pre-process the outcomes.
           Furthermore, for this measurement scheme, one should put all the outcomes into a single list and not into a different list for each Pauli.
        3. 'convert_Pauli_eigvals_to_indices' can also be used for other Pauli measurements where projection on subspace is
           involved. Note, however, that if these projectors were manually supplied in the YAML settings file, then there is
           no guarantee that the conversion will be correct.
        4. In case path to a YAML or a CSV file is provided, the file must have the extension '.yaml' or '.csv', respectively.
        # TODO: For Pauli measurements specified using YAML file, mention the order of eigenvector indices.
        # TODO: Read outcomes from a .npy file? I think this won't be very helpful.

        Arguments:
            - outcomes           : a. A list of lists/arrays, with each list/array corresponding to the outcomes of a particular POVM,
                                      listed in the same order as YAML settings file.
                                   b. A path to a YAML file (with '.yaml' extension) that lists the outcomes corresponding to each POVM.
                                   c. A path to a CSV file (with '.csv' extension) that lists the outcomes for each POVM in a separate row.
            - estimator_filename : The name of the JSON file that contains the constructed estimator.
            - estimator_dir      : Path to the directory containing the estimator file.
                                   (default: subdirectory of the current directory named 'estimator_files')
    """
    ### obtain the list of outcomes
    # a list of outcomes has been directly supplied
    if type(outcomes) in [list, tuple, np.ndarray]:
        if len(outcomes) == 0:
            raise ValueError("An empty list has been supplied.")

        # when using Randomized Pauli measurement scheme, all outcomes may be supplied in a single list
        # convert that to a list of lists for consistency with the other code
        if type(outcomes[0]) not in [list, tuple, np.ndarray]:
            outcomes_list = [np.asarray(outcomes).tolist()]
        # outcomes is already a list of lists/arrays
        else:
            outcomes_list = [np.asarray(outcome_i).tolist() for outcome_i in outcomes]
    # a path to either a YAML or a CSV file has been provided
    elif type(outcomes) == str:
        outcomes_filepath = Path(outcomes)
        if not outcomes_filepath.exists():
            raise ValueError("The specified path is invalid. Please provide a valid file path.")
        # check the file extension to determine if a YAML file or a CSV file has been provided
        file_extension = outcomes_filepath.suffix
        if file_extension == '.yaml':
            with open(outcomes_filepath) as outcomes_datafile:
                yaml_data = yaml.safe_load(outcomes_datafile)
            outcomes_list = yaml_data['outcomes']
        elif file_extension == '.csv':
            with open(outcomes_filepath) as outcomes_datafile:
                csvreader = csv.reader(outcomes_datafile, delimiter = ',')
                outcomes_list = list()
                for row in csvreader:
                    # make sure outcomes is a list of integers
                    outcomes = [int(val) for val in row]
                    outcomes_list.append(outcomes)
        else:
            raise ValueError("Only '.yaml' and '.csv' file extensions are supported.")
    # a path to a CSV file, along with additional options has been provided
    elif type(outcomes) == dict:
        try:
            outcomes_filepath = outcomes['csv_file_path']
            # specifies whether the data stored in rows or columns
            csv_entries = str(outcomes['entries']).lower()
            # index where the data starts
            csv_start = outcomes['start']
        except KeyError:
            raise ValueError("Invalid format supplied for outcomes. See docstring for help.")

        # ensure that csv_start is a tuple of non-negative integers
        if type(csv_start) in [tuple, list, np.ndarray]:
            csv_start = (int(csv_start[0]), int(csv_start[1]))
        else:
            raise ValueError("'start' must be a tuple of two of non-negative integers")

        if csv_entries == 'row':
            # the oucomes are in rows, so nothing more needs to be done
            with open(outcomes_filepath) as outcomes_datafile:
                csvreader = csv.reader(outcomes_datafile, delimiter = ',')
                outcomes_list = list()
                for (row_count, row) in enumerate(csvreader):
                    if row_count < csv_start[0]:
                        continue
                    # make sure outcomes is a list of integers
                    outcome_row = [int(val) for val in row[csv_start[1]:] if len(val) > 0]
                    outcomes_list.append(outcome_row)
        elif csv_entries == 'column':
            # the outcomes are in columns, so we need to extract the columns from the rows
            with open(outcomes_filepath) as outcomes_datafile:
                csvreader = csv.reader(outcomes_datafile, delimiter = ',')
                rows_stacked = list(csvreader)

            # consider only the rows starting from csv_start[0]
            rows_stacked = rows_stacked[csv_start[0]:]
            # number of columns in the data
            num_cols = len(rows_stacked[0]) - csv_start[1]
            if num_cols == 0:
                raise ValueError("The number of columns in the data is zero, as inferred from the options.")

            # parse the rows, and infer the columns
            csv_columns_list = [[] for _ in range(num_cols)]
            for row in rows_stacked:
                # discard the first csv_start[1] elements in the row
                row = row[csv_start[1]:]
                for index in range(num_cols):
                    val = row[index]
                    # only valid data points need to be considered
                    if len(val) > 0:
                        # add the outcome to the corresponding column, the outcomes must be integers
                        csv_columns_list[index].append(int(val))
            # the outcomes are the data stored in the columns
            outcomes_list = csv_columns_list
        else:
            raise ValueError("'entries' must either be 'row' or 'column'.")
    else:
        raise ValueError("Outcomes must be a list of lists/arrays or a path to a YAML or a CSV file.")

    ### compute the fidelity estimate
    # check that the estimator exists
    estimator_filepath = Path('%s/%s' %(estimator_dir, estimator_filename))
    if not estimator_filepath.exists():
        raise ValueError("The specified estimator file does not exist.")

    # reconstruct the estimator from the data
    with open(estimator_filepath, 'r') as data_file:
        saved_data = json.load(data_file)
        # get phi_opt_list and the constant 'c' to recreate the estimator
        phi_opt_list_data = saved_data['estimator']['phi_opt_list']
        c_data = float(saved_data['estimator']['c'])
        # get the Juditsky & Nemirovski risk of the estimator
        risk = float(saved_data['risk'])
        # check if optimization is successful
        try:
            success = bool(saved_data['success'])
        except KeyError:
            success = True
        # get the number of repetitions the estimator was constructed for
        R_list_data = saved_data['R_list']

    # a terse function to obtain an ordinal numeral from an integer
    # reference: https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
    ordinal_int = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

    # build the estimator
    def fidelity_estimator(data_list):
        """
            Given Ri independent and identically distributed elements from \Omega_i = {1, .., n_i} sampled as per p_{A^{i}(sigma)} for
            i = 1, ..., N, gives the estimate for the fidelity F(rho, sigma) = Tr(rho sigma).

            \hat{F}(\omega^{1}_1, ..., \omega^{1}_{R_1}, ... \omega^{N}_1, ..., \omega^{N}_{R_N})
                                = \sum_{i = 1}^N \sum_{l = 1}^{R_i} phi^{i}*(\omega^{i}_l) + c
        """
        N = len(data_list)

        # ensure that the risk error is not too large
        if not success:
            warnings.warn("The optimization has not converged properly to the saddle-point. The estimate cannot be trusted. Consider using a different tolerance and/or a random initial point.", MinimaxOptimizationWarning)

        # start with the terms that don't depend on the POVMs
        estimate = c_data

        # build the estimate iteratively, accounting for each POVM
        for i in range(N):
            # phi* component at the saddle point corresponding to the ith POVM
            phi_opt_i = phi_opt_list_data[i]

            # data corresponding to the ith POVM
            data_i = data_list[i]

            # ensure that only data has only Ri elements (i.e., Ri repetitions), because the estimator is built for just that case
            if not (len(data_i) == R_list_data[i]):
                warnings.warn("The number of outcomes supplied to the estimator (%d) for %s POVM does not match " %(len(data_i), ordinal_int(i))\
                               + "the number of outcomes (%s) it has been designed for. The estimate cannot be trusted." %(R_list_data[i],), MinimaxOptimizationWarning)

            estimate = estimate + np.sum([phi_opt_i[l] for l in data_i])

        return estimate

    # fidelity estimate
    F_estimate = fidelity_estimator(outcomes_list)

    # print the estimate and the risk
    print("Fidelity estimate:", np.round(F_estimate, decimals = 3))
    print("Risk:", np.round(risk, decimals = 3))

def show_yaml_file_instructions():
    """
        Prints the instructions on how to specify YAML files containing target state and measurement settings.

        TODO: density matrix vs vector for (pure) target state
              HDF5?
    """
    yaml_instructions =\
        """
            'construct_fidelity_estimator' function creates an estimator from data specified in a YAML file.
            We use a YAML file because it is human-readable, and can be parsed by Python.
            This should make it relatively convenient to specify (a possibly large number) of measurement settings.

            The target state & measurement settings can be supplied in many different ways, and the user can
            use the one that is most convenient.
            We specify the different basic formats here.

            It is important that the keys (target, POVM_list, R_list, confidence_level) in the YAML file
            are spelt as given, including the case (i.e., lower case and upper case characters should be as noted).

            ------------------------------------------------------------
            Format1: 
            ------------------------------------------------------------
            target: a list that specifies the target state as a density matrix
            POVM_list:
                - a list containing the first POVM (POVM1)
                - a list containing the second POVM (POVM2)
                .
                .
                .
            R_list: a list of number of repetitions of each POVM given in POVM_list, in that same order
            confidence_level: a value in (0.75, 1), with end-points excluded
            random_init: True or False, specifying whether a random initial condition should be used for the optimization (optional argument, default: False)
            ------------------------------------------------------------

            Notes for Format1:
                1. You cannot directly supply a numpy array to a YAML file. A list must instead be supplied.
                    A list can easily be obtained from a numpy array by using .tolist() method of the numpy array.
                    For example,
                        If data = array([[1, 2], [3, 4]]), then, you can use
                           data_as_list = data.tolist()
                        to get the data array as a list. The above returns
                            data_as_list = [[1, 2], [3, 4]]
                2. Each POVM is a list of operators.
                   For example, if POVM1 involves measurement of Pauli Z, we need to specify
                    POVM1 = [[[1, 0], [0, 0]], [[0, 0], [0, 1]]]
                   Here, the first element of POVM1, [[1, 0], [0, 0]], is the projector on first eigenbasis of Pauli Z,
                   while the second element of POVM1, [[0, 0], [0, 1]], is the projector on the second eigenbasis of Pauli Z.
                3. If a single number is provided for R_list, the same number of repetitions are used for each POVM.
                4. If there are a lot of measurement settings, it can get tedious to generate a list and then paste
                   them in the YAML file. For this purpose, Format2 can be used, which allows you to specify a file that
                   contains the data.

            ------------------------------------------------------------
            Format2 (a): 
            ------------------------------------------------------------
            target: Path to a '.npy' file that contains the numpy array specifying the target state as a density matrix.
            POVM_list:
                - Path to a '.npy' file that contains the numpy array specifying the first POVM.
                - Path to a '.npy' file that contains the numpy array specifying the second POVM.
                .
                .
                .
            R_list: a list of number of repetitions of each POVM given in POVM_list, in that same order
            confidence_level: a value in (0.75, 1), with end-points excluded
            random_init: True or False, specifying whether a random initial condition should be used for the optimization (optional argument, default: False)
            ------------------------------------------------------------

            ------------------------------------------------------------
            Format2 (b): 
            ------------------------------------------------------------
            target: Path to a '.npy' file that contains the numpy array specifying the target state as a density matrix.
            POVM_list: Path to a '.npy' file that contains a list of all POVMs as a numpy array.
            R_list: a list of number of repetitions of each POVM given in POVM_list, in that same order
            confidence_level: a value in (0.75, 1), with end-points excluded
            random_init: True or False, specifying whether a random initial condition should be used for the optimization (optional argument, default: False)
            ------------------------------------------------------------

            Notes for Format2:
                1. '.npy' is the file extension that is used by numpy.save function.
                2. If there are many POVMs that are being measured, it doesn't make sense to have many files to store each POVM.
                   Therefore, in Format2 (b), we allow to all the POVMs in one single data file as a list.
                   For example,
                       if POVM1 = [[[[1, 0], [0, 0]], [[0, 0], [0, 1]]] and POVM2 = [[[0.5, 0.5], [0.5, 0.5]], [[0.5, -0.5], [-0.5, 0.5]]],
                       then we can create a file containing the list [POVM1, POVM2], i.e., the list
                        [[[[[1, 0], [0, 0]], [[0, 0], [0, 1]]], [[[0.5, 0.5], [0.5, 0.5]], [[0.5, -0.5], [-0.5, 0.5]]]]
                       and supply the path to that file.
                3. If a single number is provided for R_list, the same number of repetitions are used for each POVM.
                4. Often, one uses Pauli measurements in experiments, and some special target states such as the W state or the stabilizer state.
                   To handle such commonly encountered scenarios, we provide some convenience in creating the YAML file. See Format3.

            ------------------------------------------------------------
            Format3: 
            ------------------------------------------------------------
            target: 
                - name_of_a_special_state: a list of parameters characterizing the state (see 'Notes for Format3' for instructions)
            POVM_list:
                - pauli: a list specifying the intended Pauli measurements (see 'Notes for Format3' for instructions)
            R_list: a list of number of repetitions of each Pauli measurement specified (see 'Notes for Format3' for instructions)
            confidence_level: a value in (0.75, 1), with end-points excluded
            random_init: True or False, specifying whether a random initial condition should be used for the optimization (optional argument, default: False)
            ------------------------------------------------------------

            Notes for Format3:
                1. The special target states that are supported, along with the format in which they need to be specified is given below.
                    a. GHZ state format        : - ghz: nq
                                                  where 'nq' denotes the number of qubits in the GHZ state.
                                                      For example, if we want to create (|000> + |111>)/sqrt(2), we specify [GHZ, 3].
                    b. W state format          : - w: nq
                                                  where 'nq' denotes the number of qubits in the W state.
                    c. Cluster state format    : - cluster: nq
                                                  where 'nq' denotes the number of qubits in the (linear) cluster state.
                    d. Werner state format     : - werner: [nq, p]
                                                  where 'nq' denotes the number of qubits in the Werner state,
                                                  and 'p' in [0, 1] characterizes the Werner state (see Wikipedia page).
                    e. Stabilizer state format : - stabilizer: generators
                                                  where generators is a 'list' of stabilizer generators for the stabilizer state.
                                                      For example, if we want to create the density matrix for the
                                                      Bell state (|00> + |11>)/sqrt(2), we can specify
                                                        - stabilizer: [XX, ZZ]
                                                      or even
                                                        - stabilizer: [XX, -YY]
                                                  Note that we need ensure that the number of Pauli operators are the same for all the generators.
                                                      For example, if we want to measure X on the third qubit for a 3-qubit stabilizer state,
                                                      we must specify IIX. Simply giving X will throw an error.
                2. Three different options are available for performing Pauli measurements, as listed below.
                    a. Pauli operators supplied as a list of strings:
                        The Pauli operators to be measured can be specified as a list of strings as follows.
                        Format:
                            - pauli: list strings of the form (sign)X_1...X_nq, with an optional argument (subspace/eigenbasis) in the list.

                        If the target state is an nq-qubit state, then each pauli operator to be measured must be a string (sign)X_1...X_nq,
                        where X_i can be any one of I, X, Y, Z, and sign can be '-' or omitted (+ need not be specified).

                        'subspace' ensures that Pauli measurements are projection on the subspace with eigenvalues +1 and -1.
                        'eigenbasis' ensures that Pauli measurements are projection on the each eigenvector of the Pauli operator.
                        These are optional arguments. The default is projection on eigenbasis.

                        For example, if we have a 2 qubit system and want to measure X & Y on each qubit with subspace projection, we can supply
                             - pauli: [XI, IX, YI, IY, subspace]
                    b. 'N' Pauli operators with highest Flammia & Liu weights:
                            Given only the number 'N' of Pauli operators to measure, chooses 'N' Pauli operators with largest Flammia & Liu weight.
                            Format:
                                - pauli: [FLS, N, eigenbasis/subspace]
                                where 'N' is the number of (non-identity) Pauli operators to be measured.
                                      eigenbasis
                       TODO: If FLS is being used, the user needs to be informed about the order in which the Pauli measurements are generated.
                    c. Randomized Pauli measurement scheme (given in section II.E., PRA):
                            Format:
                                - pauli: [RPM]

            Additional notes:
                We can mix & match elements from different formats to suit our needs.
                For example, suppose we have a target state in the form of a list called target_state_list, some other specialized measurements
                POVM_1, ..., POVM_N, also in the form of a list. In addition, we want to do Pauli measurements W1, ..., Wm. Then, we can write

                target: target_state_list
                POVM_list:
                    - POVM_1
                    .
                    .
                    .
                    - POVM_N
                    - pauli: [W1, ..., Wm]
                where W1, ..., Wm are strings specifying the required Pauli measurements.

                The only exception is when Randomized Pauli measurement scheme is used, where other measurements are not allowed.
        """
    print(yaml_instructions)
    

def parse_commandline_options():
    """
        Parses the options supplied from the command line, and calls the appropriate function to either construct the estimator or compute the estimate.

        Commandline options:
            -y or --yaml      : Path to the YAML file that contains the settings to construct the estimator
            -e or --estimator : Path to the JSON file that contains the estimator
            -o or --outcomes  : a. Path to the CSV file that contains the outcomes
                                b. A string of the form
                                        "[Path to CSV file, 'row'/'column', (row index, column index)]"
                                      OR
                                        "{'csv_file_path': Path to CSV file, 'entries': 'row'/'column', 'start': (row index, column index)}"
                                c. Path to YAML file containing the outcomes
            -q or --quiet     : If specified, the progress of optimization is not printed (optional argument, default: 0)
            -h or --help      : If specified, prints the help. Other options are ignored.

        Intended commandline usage:
            1. --yaml and --estimator are supplied             : Use the YAML settings file to construct the estimator and save it in the path specified by 'estimator'.
            2. --outcomes and --estimator are supplied         : Use the saved estimator file and specified the outcomes file to compute the fidelity estimate.
            3. --yaml, --outcomes and --estimator are supplied : Use the YAML settings file to construct the estimator, save it in the path specified by 'estimator',
                                                                 and use the outcomes file to compute the fidelity estimate.
            4. --print                                         : Optional argument that can be supplied with either 0, 1, 2, 3 above that specifies whether progress is printed.
            5. --help                                          : Prints the help. Other options are ignored.

            Any other combination raises an error.
    """
    # usage is printed at the beginning of help to show how the command is used
    usage = "Usage: python %prog [options]"
    # epilogue is printed at the end of help to describe the usage in more detail
    epilogue = "Usage of options:\n" +\
               "  1. --yaml and --estimator are supplied\n\tUse the YAML settings file to construct the estimator and save it in the path specified by 'estimator'.\n" +\
               "  2. --outcomes and --estimator are supplied\n\tUse the saved estimator file and specified the outcomes file to compute the fidelity estimate.\n" +\
               "  3. --yaml, --outcomes and --estimator are supplied\n\tUse the YAML settings file to construct the estimator, save it in the path specified by 'estimator',\n" +\
               "\tand use the outcomes file to compute the fidelity estimate.\n" +\
               "  4. --quiet\n\tOptional argument that can be supplied with 0, 1, 2, 3 to suppress printing of optimization progress.\n" +\
               "  5. --help\n\tPrints the help. Other options are ignored if help is specified.\n" +\
               "  **Any other combination will raise an error.**\n\n"
    # description of different formats to specify --outcomes
    outcomes_desc = "Formats for specifying outcomes:\n" +\
                    "  a. Path to the CSV file that contains the outcomes\n" +\
                    "\t with each row corresponding to the outcomes for a particular POVM\n".expandtabs(4) +\
                    "  b. A string of the form\n" +\
                    """\t'[Path to CSV file, "row"/"column", (row index, column index)]'\n""".expandtabs(4) +\
                    "\t\tOR\n".expandtabs(4) +\
                    """\t'{"csv_file_path": Path to CSV file, "entries": "row"/"column", "start": (row index, column index)}'\n\n""".expandtabs(4) +\
                    "\t**The single & double quotes are important, and should be as specified.**\n\tThe path must be inside double quotes.\n\n".expandtabs(4) +\
                    '\t"entries" can take the value of "row" or "column".\n'.expandtabs(4) +\
                    '\t\t- If "row", then each row must correspond to the outcomes for a particular POVM.\n'.expandtabs(4) +\
                    '\t\t- If "column", then each column must correspond to the outcomes for a particular POVM.\n'.expandtabs(4) +\
                    '\t"start" should be a tuple of non-negative integers denoting where the data starts.\n'.expandtabs(4) +\
                    """\t\t- If "start": (i, j), then the data starts at row 'i' and column 'j', in either row-wise or\n""".expandtabs(4) +\
                    '\t\t  column-wise format as specified by "entries".\n'.expandtabs(4) +\
                    "\t\t  Any entry in the CSV file before row 'i' and column 'j' is discarded.\n".expandtabs(4) +\
                    "\t\t  Note that Python is zero-indexed, so the first row/column index will be 0.\n".expandtabs(4) +\
                    "  c. Path to YAML file containing the outcomes\n\n"
    quiet_desc = "Suppressing warnings:\n" +\
                 "  --quiet can be a number between 0 & 3, with each number corresponding to different levels of suppression of printing to stdout.\n" +\
                 "\t 0: Print the progress of optimization as well as warnings to stdout.\n".expandtabs(4) +\
                 "\t 1: Suppress the progress of optimization, but print all warnings.\n".expandtabs(4) +\
                 "\t 2: Suppress the progress of optimization and MinimaxOptimizationWarning, but print other warnings.\n".expandtabs(4) +\
                 "\t 3: Suppress the progress of optimization as well as all warnings.\n".expandtabs(4) +\
                 "  MinimaxOptimizationWarning is a warning specific to the computations performed in minimax method.\n".expandtabs(4)
    epilogue += outcomes_desc + quiet_desc
    # TODO: add examples in epilogue

    # OptionParser by default strips newlines in epilog
    # so create a new class inheriting from OptionParser and change the format_epilog method
    # source: https://stackoverflow.com/questions/1857346/python-optparse-how-to-include-additional-info-in-usage-output
    class OptionParserEpilog(OptionParser):
        def format_epilog(self, formatter):
            # print the epilog "as is"
            return self.epilog

    # define a parser to parse through the commandline options
    parser = OptionParserEpilog(usage = usage, epilog = epilogue)

    # add the options that the parser should parse
    parser.add_option("-y", "--yaml", action = "store", type = "string", dest = "yaml_filepath", help = "Path to the YAML file that contains the settings to construct the estimator")
    parser.add_option("-e", "--estimator", action = "store", type = "string", dest = "estimator_filepath", help = "Path to the JSON file that contains the estimator")
    parser.add_option("-o", "--outcomes", action = "store", type = "string", dest = "outcomes", help = "Three different formats are supported. See 'Formats for specifying outcomes' for details.")
    parser.add_option("-q", "--quiet", action = "store", type = "int", dest = "quiet", default = 0, help = "Suppress the printing of progress of optimization and warning. See 'Suppressing warnings' for details.")

    # get options and arguments
    # arguments are not preceded by a hyphen; if the code requires an argument, these are compusary
    options, args = parser.parse_args()

    # this code doesn't require any arguments
    if len(args) > 0:
        raise ValueError("Positional arguments are not accepted by the code")

    # parse through supplied options
    # estimator filepath must always be supplied
    estimator_filepath = options.estimator_filepath
    if estimator_filepath is None:
        raise ValueError("--estimator must be supplied.")
    else:
        estimator_filepath = Path(estimator_filepath)

    # check if printing must be suppressed
    if options.quiet == 0:
        # print progress of optimization and warnings to stdout
        print_progress = True
    elif options.quiet == 1:
        # only supress printing of progress to stdout
        print_progress = False
    elif options.quiet == 2:
        # suppress printing of progress and MinimaxOptimizationWarning to stdout
        print_progress = False
        warnings.filterwarnings("ignore", category = MinimaxOptimizationWarning)
    elif options.quiet == 3:
        # suppress printing of progress and all warnings to stdout
        print_progress = False
        warnings.filterwarnings("ignore")
    else:
        raise ValueError("%s is an invalid entry for --quiet" %(options.quiet,))

    # one of YAML filepath or outcomes must be supplied
    yaml_filepath = options.yaml_filepath
    outcomes = options.outcomes
    if yaml_filepath is None and outcomes is None:
        raise ValueError("Either --yaml or --outcomes (or both) must be supplied.")
    
    if yaml_filepath is not None:
        yaml_filepath = Path(yaml_filepath)

        # ensure that the provided YAML file exists
        if not yaml_filepath.exists():
            raise ValueError("The specified YAML settings file does not exist. Please provide a full path to the YAML file.")

        # split the yaml filepath into filename and parent directory
        yaml_filename = yaml_filepath.name
        yaml_file_dir = yaml_filepath.parent

        # split the estimator filepath into filename and parent directory
        estimator_filename = estimator_filepath.name
        estimator_dir = estimator_filepath.parent

        # construct the estimator
        construct_fidelity_estimator(yaml_filename, estimator_filename, yaml_file_dir = yaml_file_dir, estimator_dir = estimator_dir, print_progress = print_progress)

    if outcomes is not None:
        # check whether outcomes is a valid path, else use json to parse outcomes
        if not Path(outcomes).exists():
            # parse the outcome argument using json
            try:
                outcomes = json.loads(outcomes)
            except json.JSONDecodeError:
                raise ValueError("Either a valid file path for outcomes has not been supplied or outcomes does not follow the list/dictionary format specified. Use --help to see the details.") from None

        # ensure that if a list is provided, we make a dictionary from it
        if type(outcomes) in [list, tuple]:
            if not len(outcomes) == 3:
                raise ValueError("outcomes does not follow the format specified for lists. Use --help to see the details.") from None

            outcomes = {'csv_file_path': outcomes[0], 'entries': outcomes[1], 'start': outcomes[2]}

        # ensure that the provided estimator file exists
        if not estimator_filepath.exists():
            raise ValueError("The specified estimator file does not exist. Please provide a full path to the estimator file.")

        # split the estimator filepath into filename and parent directory
        estimator_filename = estimator_filepath.name
        estimator_dir = estimator_filepath.parent

        # compute the fildeity estimator
        compute_fidelity_estimate_risk(outcomes, estimator_filename, estimator_dir = estimator_dir)

if __name__ == "__main__":
    # parse the commandline options and perform the necessary task
    parse_commandline_options()
