"""
    A compilation of some linear algebra and quantum information functions that are commonly used

    Author: Akshay Seshadri
"""

import numpy as np

import itertools

########### some linear algebra utilities
def tensor(A, B):
    """
        Uses Kroneckar product to obtain the tensor product A \otimes B of A and B

        A \otimes B = [a_{ij} B] (a pr x qs matrix if A is p x q matrix and B is r x s matrix)
    """
    AtensorB = np.concatenate([\
                    np.concatenate([A[i, j]*B for j in range(A.shape[1])], axis = 1) for i in range(A.shape[0])\
               ], axis = 0)

    return AtensorB

def get_outer_product(v, w):
    """
        Finds the outer product |v><w| for any given vectors |v> and |w>
    """
    return v.dot(np.conj(w.T))

def create_standard_basis(n, flatten = False):
    """
        Creates a list containing the elements of the standard basis for the given dimension

        |1> = (1, 0, 0, ..., 0)^T
        |2> = (0, 1, 0, ..., 0)^T
        .
        .
        .
        |n> = (0, 0, 0, ..., 1)^T
    """
    if flatten:
        first_basis_vector = np.zeros(n)
    else:
        first_basis_vector = np.zeros((n, 1))
    first_basis_vector[0] = 1.

    # the standard_basis is obtained by cyclic permutations of the first basis vector
    standard_basis = [np.array([first_basis_vector[i - j] for i in range(n)]) for j in range(n)]

    return standard_basis

def create_rank_one_projectors(n):
    """
        Creates a complete set of rank 1 projection operators for a space of dimension 'n', i.e.,
        {P_j | 1 <= j <= n} with sum_j P_j = I_n, where I_n is the identity operator for the 'n'-dimensional vector space
    """
    # since we need rank-1 projectors, we could use an orthonornal basis {|i>} of the vector space with dimension 'n'
    # that is, P_i = |i><i|; clearly, we have sum_i P_i = I_n
    # further, since the dimension of the image of any P_i is always one, each P_i is a rank-1 projector
    standard_basis = create_standard_basis(n)

    rank_one_projectors = [get_outer_product(i, i) for i in standard_basis]

    return rank_one_projectors

def embed_hermitian_matrix_real_vector_space(A, reverse = False, flatten = False):
    """
        Embeds a given Hermitian matrix A (of size n) into a real vector space of dimension n^2.

        There is an isometric isomorphism betweeen the real vector space of all n x n Hermitian matrices and the a real vector space.
        If E_ij denotes an n x n matrix with 1 at i,j and zero elsewhere, then
            E_ii                            (1 <= i <= n)
            (E_ij + E_ji)/\sqrt(2)          (1 <= i < j <= n)
            i(E_ij - E_ji)/\sqrt(2)         (1 <= i < j <= n)
        form an orthonormal basis (with respect to Frobenius inner product).

        We can then map these to the standard (orthonormal) basis e_1, ... e_{n^2} of R^{n^2}, with Euclidean inner product.
        Any linear map betweeen two orthonormal bases of finite dimensional vector spaces of same dimension is an isometric isomorphism.

        Then, the map is given as

        A -> a = ((A_ii)_i, \sqrt{2} Re(A_ij)_{i < j}, \sqrt{2} Im(A_ij)_{i < j})

        where i, j run from 1 to n.

        It is easy to verify that <A, B> = <a, b>.

        If reverse is True, then the function takes an n^2 dimensional real vector and returns an n x n Hermitian matrix.

        If the matrix is real (and symmetric) to begin with, it is enough to embed in n*(n + 1)/2 dimensional real space.
    """
    # first check if the matrix (or vector) corresponds to a symmetric or Hermitian matrix (or its embedding)
    if not reverse:
        # it is enough to check if A has any complex entries (we check the dtype instead)
        # if np.any(np.iscomplex(A)):
        if 'complex' in str(A.dtype):
            matrix_type = 'complex'
        else:
            matrix_type = 'real'
    else:
        # the input is an embedded vector, so check the dimensions
        # if the dimension is a perfect square (n^2), then the embedding is that of a Hermitian matrix
        # if not (n(n+1)/2), it is that of a symmetric matrix
        # note that n(n + 1)/2 cannot be a perfect square because if it is, there needs to be a prime number dividing both n & n + 1,
        # but gcd(n, n + 1) = 1 (any number dividing both n & n + 1 must divide their difference)
        if int(np.sqrt(A.size) + 0.5) ** 2 == A.size:
            # source: https://djangocentral.com/python-program-to-check-if-a-number-is-perfect-square/
            matrix_type = 'complex'
        else:
            matrix_type = 'real'

    if matrix_type == 'complex':
        # perform the embedding for Hermitian matrices
        # A -> a
        if not reverse:
            # if the matrix is flattened, reshape it into a square matrix
            if len(A.shape) == 1:
                n = int(np.sqrt(A.size))
                A = A.reshape((n, n))
            else:
                n = A.shape[0]

            # the vector to which the matrix is mapped
            a = np.zeros(n**2)

            ### get the components of 'a'
            # first n components are just the diagonal entries of A
            a[0:n] = np.real(np.diag(A))

            # components from n + 1 to (n^2 + n)/2 are real off-diagonal entries of A
            # components from (n^2 + n)/2 + 1 to n^2 are imaginary off-diagonal entries of A
            real_start = n                         # n is the starting point (index) for the real part
            imag_start = int(0.5*(n**2 + n))       # n + (n^2 - n)/2 is the starting point (index) for the imaginary part
            for count, (i, j) in enumerate([(i, j) for (i, j) in itertools.product(range(n), range(n)) if i < j]):
                # \sqrt{2} Re(A_ij)_{i < j}
                a[real_start + count] = np.sqrt(2)*np.real(A[i, j])

                # \sqrt{2} Im(A_ij)_{i < j}
                a[imag_start + count] = np.sqrt(2)*np.imag(A[i, j])

            return a
        # a -> A
        else:
            # write the input 'A' as 'a' following our convention
            a = A.copy()

            # size of the matrix
            n = int(np.sqrt(a.size))

            # the (Hermitian) matrix into which the vector is mapped
            A = np.zeros((n, n), dtype = 'complex128')

            # get the entries of A
            # diagonal entries
            A[np.diag_indices(n)] = a[0: n]

            # off diagonal entries
            real_start = n                         # n is the starting point (index) for the real part
            imag_start = int(0.5*(n**2 + n))       # n + (n^2 - n)/2 is the starting point (index) for the imaginary part
            for count, (i, j) in enumerate([(i, j) for (i, j) in itertools.product(range(n), range(n)) if i < j]):
                # (A_ij)_{i < j}
                A[i, j] = (a[real_start + count] + 1j*a[imag_start + count])/np.sqrt(2)

                # use Hermiticity of A
                A[j, i] = np.conj(A[i, j])

            if flatten:
                A = A.ravel()

            return A
    else:
        # perform the embedding for symmetric matrices
        # A -> a
        if not reverse:
            # if the matrix is flattened, reshape it into a square matrix
            if len(A.shape) == 1:
                n = int(np.sqrt(A.size))
                A = A.reshape((n, n))
            else:
                n = A.shape[0]

            # the vector to which the matrix is mapped
            a = np.zeros(int(n*(n + 1)/2))

            ### get the components of 'a'
            # first n components are just the diagonal entries of A
            a[0:n] = np.diag(A)

            # components from n + 1 to n(n + 1)/2 are real off-diagonal entries of A
            real_start = n                         # n is the starting point (index) for the off-diagonal part
            for count, (i, j) in enumerate([(i, j) for (i, j) in itertools.product(range(n), range(n)) if i < j]):
                # \sqrt{2} Re(A_ij)_{i < j}
                a[real_start + count] = np.sqrt(2)*np.real(A[i, j])

            return a
        # a -> A
        else:
            # write the input 'A' as 'a' following our convention
            a = A.copy()

            # size of the matrix: solve n(n + 1)/2 = a.size
            n = int((np.sqrt(1. + 8.*a.size) - 1.)/2)

            # the (Hermitian) matrix into which the vector is mapped
            A = np.zeros((n, n))

            # get the entries of A
            # diagonal entries
            A[np.diag_indices(n)] = a[0: n]

            # off diagonal entries
            real_start = n                         # n is the starting point (index) for the off-diagonal part
            for count, (i, j) in enumerate([(i, j) for (i, j) in itertools.product(range(n), range(n)) if i < j]):
                # (A_ij)_{i < j}
                A[i, j] = a[real_start + count]/np.sqrt(2)

                # use Hermiticity of A
                A[j, i] = A[i, j]

            if flatten:
                A = A.ravel()

            return A

################### sampling for generating states
def generate_uniform_points_n_sphere(n_euclid = 2, N_points = 1000, radius = 1, random_seed = False):
    """
        Generates uniform distribution of points on a n-dimensional sphere of a given radius (embedded in a Euclidean dimension n + 1).

        The centre of the sphere is assumed to be at the origin.

        Uses Marsaglia's method, which utilizes Normal distribution, to generate points on the sphere.
    """
    # generate the same sequence of random numbers if required
    if random_seed:
        np.random.seed(int(random_seed))

    # draw N_points of the form (x1, x2, ..., xn_euclid) from the normal distribution
    random_points = np.random.normal(size = (N_points, n_euclid))

    # calculate the norm of each such point: sqrt(x1**2 + x2**2 + ... + xn_euclid**2)
    random_points_norm = np.sqrt((random_points**2).sum(axis = 1))

    # stack this norm list into n_euclid number of columns placed side-by-side for easy division later
    random_points_norm_stacked = np.column_stack((random_points_norm,)*n_euclid)

    # dividing the point (x1, x2, ..., xn_euclid) by its norm gives a point on the unit sphere
    random_points_unit_sphere = random_points / random_points_norm_stacked

    # multiply these points by the necessary radius to obtain uniformly distributed points on the required sphere
    random_points_sphere = random_points_unit_sphere * radius

    return random_points_sphere

################### some QI utilities
def generate_random_state(n = 2, pure = True, density_matrix = True, flatten = True, isComplex = True, verify = False, random_seed = 1):
    """
        Generates an n-dimensional state.
        
        If a pure state is needed, either as a vector or a density matrix can be generated.

        If a mixed state is needed, a density matrix is generated.

        If flattened is true, the matrix is (row-major) flattened into a vector.

        If isComplex is true, the entries are complex numbers.

        If verify is true, a check is performed to ensure that the generated matrix is indeed a density matrix.
    """
    if random_seed:
        np.random.seed(int(random_seed))

    if pure:
        # rho is a pure state, which we randomly generate
        if isComplex:
            rho = generate_uniform_points_n_sphere(n_euclid = 2*n, N_points = 1, random_seed = random_seed)[0]
            rho = np.array([p1 + 1.j*p2 for (p1, p2) in zip(rho[0::2], rho[1::2])]) # create a state vector with complex coefficients
        else:
            rho = generate_uniform_points_n_sphere(n_euclid = n, N_points = 1, random_seed = random_seed)[0]
        rho = rho.reshape((n, 1))
        if density_matrix:
            if flatten:
                rho = get_outer_product(rho, rho).ravel()               # density matrix corresponding to the state vector; flattened
            else:
                rho = get_outer_product(rho, rho)                       # density matrix corresponding to the state vector
    else:
        # construct a matrix sampled from the Ginibre ensemble (entries are iid complex standard normal)
        G = np.random.randn(n, n) + 1j*np.random.randn(n, n)

        # construct a density matrix as per the Hilbert-Schmidt ensemble (arXiv:1010.3570)
        rho = G.dot(np.conj(G.T))
        rho = rho / np.trace(rho)

        if flatten:
            rho = rho.ravel()

    if verify:
        # write rho as a density matrix in unflattened form
        rho = rho.reshape((n, n))

        # check that it is Hermitian
        if not np.linalg.norm(np.conj(rho.T) - rho) < 1e-8:
            raise ValueError("The generated matrix is not Hermitian")

        # check that it is positive sem-definite
        rho_eigvals = np.real(np.linalg.eigvalsh(rho))
        if np.any(rho_eigvals) < 0:
            raise ValueError("The generated matrix is not positive semi-definite")

        # check that the eigenvalues sum to one (equivalently, the trace is one)
        if not np.abs(sum(rho_eigvals) - 1) < 1e-8:
            raise ValueError("The trace of the generated matrix is not 1.")

        # reflatten if necessary
        if flatten:
            rho = rho.ravel()

    return rho

def generate_special_state(state = 'GHZ', state_args = {'d': 2, 'M': 2}, density_matrix = True, flatten = True, isComplex = True):
    """
        Generates special types of quantum states used in quantum information tasks.

        Each state is given its separate set of arguments which are described below.
        state: which state to generate. Valid options are (lower case also accepted)
                 - 'GHZ'
                 - 'W'
                 - 'cluster'
                 - 'Werner' (Note: Werner state is a density matrix instead of a vector by definition, so density_matrix is ignored)
                 - 'stabilizer' (Note: only a density matrix version of the stabilizer state is currently supported, so density_matrix = False
                                       will raise an error. It is also not verified if the list of Pauli operators specified are valid generators.)
        state_args: a dictionary of arguments specific to each state
                    - GHZ: {'d', 'M'} where 'd' is the dimension of each subsystem (qudit),
                                        and 'M' is the number of copies of qudits
                    - W: {'nq'} where 'nq' is the number of qubits
                    - cluster: {'nq'} where 'nq' is the number of qubits
                    - Werner: {'nq', 'p'} where 'nq' is the number of qubits, and 'p' \in [0, 1] is the parameter
                              characterizing the Werner state
                    - stabilizer: {'nq', 'generators'} where 'nq' is the number of qubits,
                                  and 'generators' are a list of 'nq' Pauli strings that generate the stabilizer state.
                                  If the Pauli operator has a phase of -1, then a minus sign should preceed the Pauli string.

        dentisty_matrix: if True, gives |state><state|
        flatten: if True, density matrix is flattened (in row-major fashion)
    """
    # check that a valid state has been requested
    if not str(state).lower() in ['ghz', 'w', 'cluster', 'werner', 'stabilizer']:
        raise ValueError("%s is not a valid argument for state. See function docstring for details." %state)
    else:
        state = str(state).lower()

    # a dictionary of arguments for every state
    state_args_dict = {'ghz': {'d': -1, 'M': -1}, 'w': {'nq': -1}, 'cluster': {'nq': -1}, 'werner': {'d': -1, 'p': -1},\
                       'stabilizer': {'nq': -1, 'generators': []}}

    # check that valid state arguments have been provided
    if not (type(state_args) == dict and state_args.keys() == state_args_dict[state].keys()):
        raise ValueError("Please provide valid parameters to generate the state. See function docstring for details.")

    if state == 'ghz':
        d = int(state_args['d'])
        M = int(state_args['M'])

        # the ghz state for qubits is particularly simple (in computational basis), so handle that separately
        if d == 2:
            generated_state = np.zeros(d**M)
            generated_state[0] = 1.
            generated_state[-1] = 1.
        else:
            # basis for qudits: |0>, ..., |d - 1>
            qudit_basis = create_standard_basis(d, flatten = True)
            # M-fold tensor product of each basis vector: |ii...i> (M -times) for i = 1, ..., M
            # GHZ is sum of such M-fold tensor product of basis vectors, normalized
            generated_state = np.zeros(d**M)
            for (i, basis_elt) in enumerate(qudit_basis):
                Mfold_tensor_qudit_basis = basis_elt
                for _ in range(1, M):
                    Mfold_tensor_qudit_basis = np.kron(Mfold_tensor_qudit_basis, basis_elt)

                generated_state += Mfold_tensor_qudit_basis

        # normalize
        generated_state = generated_state / np.sqrt(d)
    elif state == 'w':
        nq = int(state_args['nq'])

        # the w state is one-excitation state: (|10..0> + ... + |0...01>)/sqrt(nq))
        ground_state, excited_state = create_standard_basis(2, flatten = True)
        generated_state = np.zeros(2**nq)
        for i in range(nq):
            # tensor of i - 1 ground states, 1 excited state, then nq - i ground state
            excitation = np.ones(1)
            for j in range(i):
                excitation = np.kron(excitation, ground_state)
            excitation = np.kron(excitation, excited_state)
            for j in range(i + 1, nq):
                excitation = np.kron(excitation, ground_state)
            generated_state += excitation

        # normalize
        generated_state = generated_state / np.sqrt(nq)
    elif state == 'cluster':
        nq = int(state_args['nq'])

        # cluster state can be created by applying Hadamard to |0...0> and 
        # then Controlled-Z on adjacent qubits
        generated_state = np.zeros(2**nq)
        generated_state[0] = 1.
        # |+...+> can be obtained by applying Hadamard to each qubit
        H = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
        H_nq = np.eye(1)
        for i in range(nq):
            H_nq = np.kron(H_nq, H)
        generated_state = H_nq.dot(generated_state)

        # create Controlled-Z acting on adjacent qubits
        if nq > 1:
            CZ = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0], [0, 0, 0, -1.]])
            CZ_adjacent = np.eye(2**nq)
            for i in range(nq - 1):
                CZ_nq = np.eye(1)
                for j in range(i):
                    CZ_nq = np.kron(CZ_nq, np.eye(2))
                CZ_nq = np.kron(CZ_nq, CZ)
                for j in range(i + 2, nq):
                    CZ_nq = np.kron(CZ_nq, np.eye(2))
                CZ_adjacent = CZ_adjacent.dot(CZ_nq)

            generated_state = CZ_adjacent.dot(generated_state)
    elif state == 'werner':
        # Werner state as defined in Wikipedia is generated here
        # also see https://arxiv.org/pdf/1402.2413.pdf, but the trace there is 2 instead of 1
        # Werner state is a density matrix that lives in C^d \otimes C^d
        d = int(state_args['d'])

        # parameter characterizing the Werner state
        p = float(state_args['p'])
        if not (0 <= p <= 1):
            raise ValueError("The parameter 'p' characterizing the Werner state must be between 0 and 1")

        # flip operator
        F = generate_SWAP_operator(d, flatten = False)

        # projection on to the symmetric subspace: I_d \otimes I_d = I_{d^2}
        P_symm = 0.5*(np.eye(d**2) + F)
        # projection on to the anti-symmetric subspace
        P_asymm = 0.5*(np.eye(d**2) - F)

        # the Werner state
        generated_state = p * 2*P_symm / (d*(d + 1)) + (1 - p) * 2*P_asymm / (d*(d - 1))

        # since Werner state is already a density matrix, no need to convert it into one
        density_matrix = False

        # flatten if necessary
        if flatten:
            generated_state = generated_state.ravel()
    elif state == 'stabilizer':
        # an nq-qubit stabilizer state is uniquely specified from nq Pauli generators
        nq = int(state_args['nq'])

        # Pauli operator strings that generate the state
        generators = state_args['generators']

        # make sure that Pauli strings in generators are listed using 0, 1, 2, 3 instead of I, X, Y, Z
        generators = [g.lower().translate(str.maketrans('ixyz', '0123')) for g in generators]

        # separate the phase from the Pauli operators in generators; the phase for stabilizers can only be +1 or -1
        generators_pauli = [generator.strip('-').strip('+') for generator in generators]
        generators_phase = [-1 if '-' in generator else 1 for generator in generators]

        # generate the Pauli operators specified by the generators, with phase included
        pauli_operator_list = [phase*pauli_operator for (pauli_operator, phase) in zip(generate_Pauli_operator(nq = nq, index_list = generators_pauli, flatten = False), generators_phase)]

        # the stabilizer is given by the product of projectors on the +1 eigensubspaces of the Pauli generators
        # the Paulis (and therefore the projectors) commute, so the order does not matter
        generated_state = np.eye(2**nq)
        for pauli_operator in pauli_operator_list:
            # projector on +1 eigensubspace of generator G is (I + G) / 2 (phase must already be accounted for in G)
            generated_state = 0.5*generated_state.dot(np.eye(2**nq) + pauli_operator)

        if not density_matrix:
            raise ValueError("Only stabilizer states in the form a density matrix are currently supported")

        # since the stabilizer state is already a density matrix, no need to convert it into one
        density_matrix = False

        # flatten if necessary
        if flatten:
            generated_state = generated_state.ravel()

    if density_matrix:
        generated_state = generated_state.reshape((generated_state.size, 1))
        generated_state = get_outer_product(generated_state, generated_state)

        if flatten:
            generated_state = generated_state.ravel()

    if isComplex:
        generated_state = generated_state.astype('complex128')

    return generated_state

def generate_POVM(n = 2, num_povm = 2, projective = False, flatten = True, isComplex = True, verify = False, pauli = None, random_seed = 1):
    """
        Generates a POVM with 'num_povm' elements. Each POVM element is of size n x n (if flatten is true, then n^2).
        A POVM is defined by a set of 'num_povm' positive semi-definite operators that sum to identity.

        If projective measurements are required, then each POVM element is of rank 1 (i.e., a projector). For projective measurement,
        'num_povm' is equal to 'n' (so the 'num_povm' argument is ignored).
        For Pauli measurements, if num_povm is 2, then POVM elements are projectors on to the subspaces with +1, -1 eigenvalue.
        And if num_povm is n, then the POVM elements are rank 1 projectors on to each eigenvector of the given Pauli operator.

        If 'projective' is true and 'pauli' is None, projectors corresponding to the the standard basis is returned.
        
        If 'projective' is true, 'n' is a power of 2 and 'pauli' is a string denoting a Pauli operator, 
        then a projective measurement corresponding to the specified Pauli matrix is returned.
            - The pauli string should be of the form 'i_1i_2...i_nq', where each i_k is between 0 & 3 which refers to the Pauli operator
              \sigma[i_1] \otimes \sigma[i_2] \otimes ... \otimes \sigma[i_nq].
            - The isComplex argument is ignored if pauli is specified (the returned POVM always has the dtype 'complex128').

        If isComplex is true, the entries are complex numbers (even if they are real, the type is changed to complex).

        If verify is true, a check is performed to ensure that the generated matrices indeed form a POVM.

        TODO: 1. Move 'pauli' argument to after 'projective' and before 'flatten'; not done now to avoid breaking any other code that may be supplying
                 positional arguments (instead of keyword arguments) to generate_POVM function.
              2. Add support for specifying phase (+/-1, +/-i) as a part of pauli.
    """
    if projective:
        if pauli is None or pauli is False:
            num_povm = n
            POVM = create_rank_one_projectors(n)                # create projective measurements
        else:
            # number of qubits
            nq = int(np.log2(n))
            if 2**nq != n:
                raise ValueError("Pauli measurement possible only in systems of qubits, i.e., the dimension should be a power of 2")

            # generate POVM depending on whether projectors on subpace or projectors on each eigenvector is required
            if num_povm == 2:
                # the Pauli operator that needs to be measured
                Pauli_operator = generate_Pauli_operator(nq, pauli)[0]

                # if W is the Pauli operator and P_+ and P_- are projectors on to the eigenspaces corresponding to +1 & -1 eigenvalues, then
                # P_+ - P_- = W, and P_+ + P_- = \id. We can solve for P_+ and P_- from this.
                P_plus  = 0.5*(np.eye(n, dtype = 'complex128') + Pauli_operator)
                P_minus = 0.5*(np.eye(n, dtype = 'complex128') - Pauli_operator)

                POVM = [P_plus, P_minus]
            elif num_povm == n:
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
                POVM = [transform_matrix.dot(Ei).dot(np.conj(transform_matrix.T)) for Ei in computational_basis_POVM]
            else:
                raise ValueError("Pauli measurements with only 2 or 'n' POVM elements are supported")
    else:
        if num_povm > n:
            raise ValueError("Number of measurements can at most be 'n'")

        if random_seed:
            np.random.seed(int(random_seed))

        # generate num_povm positive semi-definite matrices that sum to identity
        POVM = list()
        for i in range(num_povm):
            if isComplex:
                Ei = np.random.randn(n, n) + 1j*np.random.randn(n, n)
            else:
                Ei = np.random.randn(n, n)
            Ei = np.conj(Ei.T).dot(Ei)                              # obtain a positve semi-definite matrix
            POVM.append(Ei)

        # to generate a POVM, we need the operators to sum to identity
        # If S = sum_i E_i, then we can redefine E_i = S^{-1/2} E_i S^{-1/2}, so that sum_i E_i = I (see Definiion 5.8 of
        # 'Random Positive Operator Valued Measures' (arxiv:1902.04751))
        S = np.sum(POVM, axis = 0)
        # now obtain S^{-1/2} through spectral decomposition
        S_eigvals, S_eigvecs = np.linalg.eigh(S)
        S_minushalf_eigvals  = S_eigvals**(-0.5)
        S_minushalf = S_eigvecs.dot( np.diag(S_minushalf_eigvals).dot( S_eigvecs.T.conj() ) )

        # create the POVM from Ei and S^{-1/2}
        POVM = [S_minushalf.dot( Ei.dot( S_minushalf ) ) for Ei in POVM]

    # flatten the POVM elements, if required
    if flatten:
        POVM = [Ei.ravel() for Ei in POVM]

    # make sure that the entries in all the POVM elements are interpreted as complex numbers, if required
    if isComplex:
        POVM = [Ei.astype('complex128') for Ei in POVM]

    # verify that the elements of the POVM sum to identiy
    if verify:
        if flatten:
            eye_n = np.eye(n).ravel()
        else:
            eye_n = np.eye(n)

        if not np.linalg.norm(np.sum(POVM, axis = 0) - eye_n) < 1e-6:
            raise ValueError("Not a POVM")

    return POVM

def generate_Pauli_operator(nq = 1, index_list = '3', flatten = False):
    """
        Given the number of qubits 'nq', generates the Pauli operators corresponding to each index in 'index_list'.

        An index can be a string (of length 'nq') or an integer (between 0 & 4^nq - 1) that denotes which Pauli operator to generate.
        The following convention is used.

        Pauli operators:
        \sigma[0] = \id, \sigma[1] = \sigma_x, \sigma[2] = \sigma_y, \sigma[3] = \sigma_z

        If index is a string: 
            index = 'i_1i_2...i_nq', where each i_k is between 0 & 3, then the Pauli operator
            \sigma[i_1] \otimes \sigma[i_2] \otimes ... \otimes \sigma[i_nq] is generated.

        If the index is an integer between 0 & 4^nq - 1, the following bijective mapping is used to obtain a string: 
            'i_1i_2...i_nq' = i_1*4^{nq - 1} +  i_2*4^{nq - 2} + ... + i_nq
            (base 4 <-> base 10 conversion)

        All operators are written in the eigenbasis of \sigma[3] \otimes ... \otimes \sigma[3].

        A Pauli operator of acting on 2^nq-dimensional space is returned.
    """
    # the standard 2x2 Pauli operators
    sigma = [np.eye(2, dtype = 'complex128'), np.array([[0., 1.], [1., 0.]], dtype = 'complex128'),\
             np.array([[0., -1j], [1j, 0.]], dtype = 'complex128'), np.array([[1., 0.], [0., -1.]], dtype = 'complex128')]

    if type(index_list) != list:
        index_list = [index_list]

    pauli_list = [0]*len(index_list)

    for count, index in enumerate(index_list):
        if type(index) in [int, np.int64]:
            if index > 4**nq - 1:
                raise ValueError("Each index must be a number between 0 and 4^{nq} - 1")
            # make sure index is a string
            index = np.base_repr(index, base = 4)
            # pad the index with 0s on the left so that the total string is of size nq (as we need a Pauli operator acting on nq qubits)
            index = index.rjust(nq, '0')
        elif type(index) == str:
            # get the corresponding integer
            index_num = np.array(list(index), dtype = 'int')
            index_num = index_num.dot(4**np.arange(len(index) - 1, -1, -1))
            
            if index_num > 4**nq - 1:
                raise ValueError("Each index must be a number between 0 and 4^{nq} - 1")

            # pad the index with 0s on the left so that the total string is of size nq (as we need a Pauli operator acting on nq qubits)
            index = index.rjust(nq, '0')

        # use the fact that tensor products are associative to recursively build the Pauli operator corresponding to the index
        # A_1 \otimes A_2 \otimes ... \otimes A_n = (A_1 \otimes (A_2 \otimes ... (A_{n - 1} \otimes A_n)...))
        pauli_operator = sigma[int(index[-1])]
        for i in index[-2::-1]:
            pauli_operator = np.kron(sigma[int(i)], pauli_operator)

        if flatten:
            pauli_operator = pauli_operator.ravel()

        pauli_list[count] = pauli_operator

    return pauli_list

def generate_SWAP_operator(n, flatten = False, isComplex = True):
    """
        Geneates the SWAP (or flip) operator 'F' defined as
        F(|\psi> \otimes |\phi>) = |\phi> \otimes |\psi>
        where |\psi>, |\phi> \in C^n.

        Given a basis {|i>} of C^n, we can write
        F = \sum_{i, j} |i><j| \otimes |j><i|
        It can be shown that this is basis independent (see example 3.1 in https://arxiv.org/pdf/1402.2413.pdf).
    """
    flip_operator = np.zeros((n**2, n**2))

    # the matrix |i><j| \otimes |i'><j'| has a 1 at the index ((num of rows of |i'><j'|) * i + i', (num of cols of |i'><j'|) * j + j')
    for (i, j) in itertools.product(range(n), range(n)):
        flip_operator[n*i + j, n*j + i] = 1

    if flatten:
        flip_operator = flip_operator.ravel()

    if isComplex:
        flip_operator = flip_operator.astype('complex128')

    return flip_operator
