# Minimax method for fidelity estimation

This package can be used for fidelity estimation using the minimax method.

The module `handle_fidelity_estimation.py` provides a "frontend" for using the package. This code can be used to implement the minimax method from either the Python console or from a terminal.\
The code makes it easy to specify the target state and measurement settings, in order to estimate the fidelity.

For a quick introduction, check the examples given in the IPython notebook `fidelity_estimation_examples.ipynb`.

# Installation and Dependencies
The code can be used "out of the box". No installation is necessary.\
We list the dependencies that must be installed for using the code.\
The version number using which the code was tested is written in brackets, next to the package.

- python (3.8.1)
- ipython (7.12.0)
- yaml (0.1.7)
- numpy (1.18.1)
- scipy (1.4.1)
- matplotlib (3.1.3)
- jupyter-notebook

One can install these packages in a virtual environment.\
The file `dependencies/dependencies.yml` contains a list of packages that can be used to recreate the environment used to test the code.\
Install [Anaconda/Miniconda](https://conda.io/projects/conda/en/latest/index.html) and run `conda env create -f dependencies.yml' in the folder containing `dependencies.yml` file.
This will create an environment with the name `minimax_fidelity`. To start using this environment, run `conda activate minimax_fidelity`.

# Usage
The code can be interactively via the Python/IPython console or through the commandline. We provide instructions for both these.

The typical workflow is as follows:
1. Specify the target state, measurement settings, and the confidence level through a YAML file. ([YAML](https://pyyaml.org/) is a human-readable markup language, which can be parsed by a computer.)
2. Create an estimator for fidelity using `handle_fidelity_estimation.py` and the YAML file. This estimator is stored in a [JSON](https://www.json.org/json-en.html) file, and can be reused in the future for the same settings.
3. Supply the outcomes to `handle_fidelity_estimation.py` along with the JSON file for the estimator to obtain the fidelity estimate and the risk.

We will postpone the details on how to specify the settings using a YAML file to [YAML settings file](#yaml) section.

## <a name"interactive">Interactive usage:
We provide two functions for interactive usage, one for constructing the estimator and another for generating an estimate from the outcomes.\
These functions can be imported into a Python interpreter from the `handle_fildeity_estimation.py` module.

### Constructing the estimator:
The estimator can be constructed using the function `construct_fidelity_estimator`. The syntax is as follows.

```
construct_fidelity_estimator(yaml_filename, estimator_filename,\
                             yaml_file_dir = './yaml_files', estimator_dir = './estimator_files',\
                             print_progress = True)
```

#### Options:
|----------------------|----------------------------------------|
| :---:                | :---                                   |
| `yaml_filename` | Name of the YAML settings file |
| `estimator_filename` | Name of the file to which the estimator will be saved after it is constructed |
| `yaml_file_dir` | Name of the directory containing the YAML settings file. The defult value is the sub-directory `yaml_files` of the project (root) directory. This sub-directory must be created beforehand. |
| `estimator_dir` | Name of the directory to which the estimator file will be saved. The defult value is the sub-directory `estimator_files` of the project (root) directory. This sub-directory is created if it doesn't exist. |
| `print_progress` | Specify whether the progress of optimization is printed |
|----------------------|----------------------------------------|

---------
Note: The risk is computed along with the estimator, *before* the outcomes are supplied. This risk is printed along with the estimate when outcomes are supplied. This risk is stored in the JSON file containing the estimator.
---------

### Estimating the fidelity from outcomes:
The estimator can be constructed using the function `construct_fidelity_estimate_risk`. The syntax is as follows.

```
compute_fidelity_estimate_risk(outcomes, estimator_filename, estimator_dir = './estimator_files')
```

#### Options:
|----------------------|----------------------------------------|
| :---:                | :---                                   |
| `outcomes` | The outcomes can be supplied to the function in three different ways:\
                1. A list of lists/arrays, with each list/array corresponding to the outcomes of a particular POVM, listed in the same order as YAML settings file.
                2. A path to a YAML file (with `.yaml` extension) that lists the outcomes corresponding to each POVM.
                3. A path to a CSV file (with `.csv` extension) that lists the outcomes for each POVM in a separate row. More customization is possible.
                A detailed description of how to specify outcomes is given in section [Specifying outcomes](#outcomes) |
| `estimator_filename` | Name of the JSON file (with `.json` extension) containing the constructed estimator. |
| `estimator_dir` | Name of the directory containing the estimator file. The defult value is the sub-directory `estimator_files` of the project (root) directory. |
|----------------------|----------------------------------------|

## Commandline usage:
The `handle_fidelity_estimation.py` module can also be used from the commandline. This is particularly useful when the code needs to be run remotely on a cluster or a workstation.

```
python handle_fidelity_estimation.py [options]
```

#### Options:
|----------------|--------------------------------------------------------------------------------------------------------------|
|-y, --yaml      | Path to the YAML file that contains the settings to construct the estimator |
|-e, --estimator | Path to the JSON file that contains the estimator |
|-o, --outcomes  | a. Path to the CSV file that contains the outcomes |
|                | b. A string of the form\
                         "[Path to CSV file, 'row'/'column', (row index, column index)]"\
                       OR\
                         "{'csv_file_path': Path to CSV file, 'entries': 'row'/'column', 'start': (row index, column index)}"\
                    *The single & double quotes are important, and should be as specified. The path must be inside double quotes.*
                    - "entries" can take the value of "row" or "column"
                      - If "row", then each row must correspond to the outcomes for a particular POVM.
                      - If "column", then each column must correspond to the outcomes for a particular POVM.
                    - "start" should be a tuple of non-negative integers denoting where the data starts.
                      - If "start": (i, j), then the data starts at row 'i' and column 'j', in either row-wise or column-wise format as specified by "entries".
                      - Any entry in the CSV file before row 'i' and column 'j' is discarded.
                      - Note that Python is zero-indexed, so the first row/column index will be 0. |
|                | c. Path to YAML file containing the outcomes |
|-q, --quiet     | If specified, the progress of optimization is not printed (optional argument, default: 0)\
                   --quiet can be a number between 0 & 3, with each number corresponding to different levels of suppression of printing to stdout.
                   - 0: Print the progress of optimization as well as warnings to stdout
                   - 1: Suppress the progress of optimization, but print all warnings
                   - 2: Suppress the progress of optimization and MinimaxOptimizationWarning, but print other warnings
                   - 3: Suppress the progress of optimization as well as all warnings 
                   *MinimaxOptimizationWarning is a warning specific to the computations performed in minimax method.* |
|-h, --help      | If specified, prints the help. Other options are ignored. |
|----------------|--------------------------------------------------------------------------------------------------------------|

### Intended commandline usage:
|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| --yaml and --estimator are supplied             | Use the YAML settings file to construct the estimator and save it in the path specified by 'estimator'. |
| --outcomes and --estimator are supplied         | Use the saved estimator file and specified the outcomes file to compute the fidelity estimate. |
| --yaml, --outcomes and --estimator are supplied | Use the YAML settings file to construct the estimator, save it in the path specified by 'estimator',and use the outcomes file to compute the fidelity estimate. |
| --print                                         | Optional argument that can be supplied with either 0, 1, 2, 3 above that specifies whether progress is printed. |
| --help                                          | Prints the help. Other options are ignored. |
|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------|

Any other combination will raise an error.

**Note**: The commandline utility has only been tested on a Linux machine using a BASH shell.

# <a name="yaml">YAML settings file
The target state & measurement settings can be supplied in many different ways, and the user can use the one that is most convenient.\
We provide three basic formats that can be used to YAML file. One can mix and match elements from these different formats as necessary.

**Note**: It is important that the keys (target, POVM_list, R_list, confidence_level) in the YAML file are spelt as given, including the case (i.e., lower case or upper case) of the characters.

## Format 1: 
target: a list that specifies the target state as a density matrix\
POVM_list:\
    \- a list containing the first POVM (POVM1)\
    \- a list containing the second POVM (POVM2)\
    .\
    .\
    .\
R_list: a list of number of repetitions of each POVM given in POVM_list, in that same order\
confidence_level: a value in (0.75, 1), with end-points excluded\
random_init: True or False, specifying whether a random initial condition should be used for the optimization (optional argument, default: False)

------------------------------------------------------------
Notes for Format 1:
1. You cannot directly supply a numpy array to a YAML file. A list must instead be supplied.\
    A list can easily be obtained from a numpy array by using `.tolist()` method of numpy array.\
    - For example, if `data = array([[1, 2], [3, 4]])`, then you can use `data_as_list = data.tolist()` to get the data array as a list.\
      This returns `data_as_list = [[1, 2], [3, 4]]`.
2. Each POVM is a list of operators.
   - For example, if POVM1 involves measurement of Pauli Z, we need to specify `POVM1 = [[[1, 0], [0, 0]], [[0, 0], [0, 1]]]`.\
     Here, the first element of `POVM1`, `[[1, 0], [0, 0]]`, is the projector on first eigenbasis of Pauli Z,
     while the second element of `POVM1`, `[[0, 0], [0, 1]]`, is the projector on the second eigenbasis of Pauli Z.
3. If a single number is provided for `R_list`, the same number of repetitions are used for each POVM.
4. If there are a lot of measurement settings, it can get tedious to generate a list and then paste
   them in the YAML file. For this purpose, `Format 2` can be used, which allows you to specify a file that contains the data.
------------------------------------------------------------

## Format 2: 
### Format 2(a)
target: Path to a `.npy` file that contains the numpy array specifying the target state as a density matrix.\
POVM_list:\
    \- Path to a `.npy` file that contains the numpy arrays specifying the first POVM.\
    \- Path to a `.npy` file that contains the numpy arrays specifying the second POVM.\
    .\
    .\
    .\
R_list: a list of number of repetitions of each POVM given in POVM_list, in that same order\
confidence_level: a value in (0.75, 1), with end-points excluded\
random_init: True or False, specifying whether a random initial condition should be used for the optimization (optional argument, default: False)

### Format 2(b): 
target: Path to a `.npy` file that contains the numpy array specifying the target state as a density matrix.\
POVM_list: Path to a `.npy` file that contains a list of all POVMs (each POVM is stored as a list of numpy arrays).\
R_list: a list of number of repetitions of each POVM given in POVM_list, in that same order\
confidence_level: a value in (0.75, 1), with end-points excluded\
random_init: True or False, specifying whether a random initial condition should be used for the optimization (optional argument, default: False)

------------------------------------------------------------
Notes for Format2:
    1. `.npy` is the file extension that is used by `numpy.save` function.
    2. If there are many POVMs that are being measured, it doesn't make sense to have many files to store each POVM.\
       Therefore, in Format 2(b), we allow to all the POVMs in one single data file as a list.
       - For example, if `POVM1 = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]` and `POVM2 = [np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0.5, -0.5], [-0.5, 0.5]]])`,\
         then we can create a file containing the list `[POVM1, POVM2]`, i.e., the list\
         `[[np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])], [np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0.5, -0.5], [-0.5, 0.5]])]]`\
         and supply the path to that file.
    3. If a single number is provided for `R_list`, the same number of repetitions are used for each POVM.
    4. Often, one uses Pauli measurements in experiments, and some special target states such as the W state or the stabilizer state.\
       To handle such commonly encountered scenarios, we provide some convenience in creating the YAML file. See `Format 3`.
------------------------------------------------------------

## Format 3: 
target:\ 
    \- name_of_a_special_state: a list of parameters characterizing the state (see "Notes for `Format 3`" for instructions)\
POVM_list:\
    \- pauli: a list specifying the intended Pauli measurements (see "Notes for `Format 3`" for instructions)\
R_list: a list of number of repetitions of each Pauli measurement specified (see "Notes for `Format 3`" for instructions)\
confidence_level: a value in (0.75, 1), with end-points excluded\
random_init: True or False, specifying whether a random initial condition should be used for the optimization (optional argument, default: False)

------------------------------------------------------------
Notes for Format3:
    1. The special target states that are supported, along with the format in which they need to be specified is given below.
        |-------------------------|--------------------------------------------------------------------------------------|
        | GHZ state format        | \- ghz: nq
                                      where 'nq' denotes the number of qubits in the GHZ state.\
                                      For example, if we want to create (|000> + |111>)/sqrt(2), we specify [GHZ, 3]. |
        | W state format          | \- w: nq
                                      where 'nq' denotes the number of qubits in the W state. |
        | Cluster state format    | \- cluster: nq
                                      where 'nq' denotes the number of qubits in the (linear) cluster state. |
        | Werner state format     | \- werner: [nq, p]
                                      where 'nq' denotes the number of qubits in the Werner state,
                                      and 'p' in [0, 1] characterizes the Werner state. |
        | Stabilizer state format | \- stabilizer: generators
                                      where generators is a 'list' of stabilizer generators for the stabilizer state.\
                                     - For example, if we want to create the density matrix for the
                                       Bell state (|00> + |11>)/sqrt(2), we can specify\
                                            \- stabilizer: [XX, ZZ]\
                                          or even\
                                            \- stabilizer: [XX, -YY]\
                                      Note that we need ensure that the number of Pauli operators are the same for all the generators.\
                                      - For example, if we want to measure X on the third qubit for a 3-qubit stabilizer state,
                                        we must specify IIX. Simply giving X will throw an error. |
        |----------------------------------------------------------------------------------------------------------------|
    2. Three different options are available for performing Pauli measurements, as listed below.
        - Pauli operators supplied as a list of strings:\
            The Pauli operators to be measured can be specified as a list of strings as follows.\
            Format:\
                \- pauli: list strings of the form (sign)X_1...X_nq, with an optional argument (subspace/eigenbasis) in the list.

            If the target state is an nq-qubit state, then each pauli operator to be measured must be a string (sign)X_1...X_nq,
            where X_i can be any one of I, X, Y, Z, and sign can be '-' or omitted (+ need not be specified).

            `subspace` ensures that Pauli measurements are projection on the subspace with eigenvalues +1 and -1.\
            `eigenbasis` ensures that Pauli measurements are projection on the each eigenvector of the Pauli operator.\
            These are optional arguments. The default is projection on eigenbasis.

            - For example, if we have a 2 qubit system and want to measure X & Y on each qubit with subspace projection, we can supply\
                 \- pauli: [XI, IX, YI, IY, subspace]
        - `N` Pauli operators with highest weights as per DFE:\
            Given only the number 'N' of Pauli operators to measure, chooses 'N' Pauli operators with largest DFE weights `|Tr(target Pauli)|`.\
            Format:\
                \- pauli: [FLS, N, eigenbasis/subspace]\
                where 'N' is the number of (non-identity) Pauli operators to be measured.\
            TODO: If FLS is being used, the user needs to be informed about the order in which the Pauli measurements are generated.
        - Randomized Pauli measurement scheme (given in section II.E., PRA submission):\
            Format:\
                \- pauli: [RPM]\
------------------------------------------------------------

## Additional notes:
We can mix & match elements from different formats to suit our needs.\
- For example, suppose we have a target state in the form of a list called `target_state_list`, some other specialized measurements
  `POVM_1`, ..., `POVM_N`, also in the form of a list. In addition, we want to do Pauli measurements `W1`, ..., `Wm`. Then, we can write\

  target: target_state_list\
  POVM_list:\
      \- POVM_1\
      .\
      .\
      .\
      \- POVM_N\
      \- pauli: [W1, ..., Wm]\
  where `W1`, ..., `Wm` are strings specifying the required Pauli measurements.

The only exception is when Randomized Pauli measurement scheme is used, where other measurements are not allowed.

# <a name="outcomes">Specifying outcomes
The outcomes argument in `compute_fidelity_estimate_risk` function (see [interactive usage](#interactive)) can be specified in on of the following formats:
1. A list of arrays (or lists), with each array corresponding to the outcomes for a particular POVM measurement.
2. Path to a YAML file in the following format:\
    outcomes:\
        \- outcomes for POVM 1\
        \- outcomes for POVM 2\
        .\
        .\
        .
3. CSV files in one of the following formats:\
    - Path to a CSV file with each row corresponding to the outcomes for a particular POVM.
    - A dictionary `{'csv_file_path': Path to CSV file, 'entries': 'row', 'start': (row index, column index)}`
        - `entries` can take the value of 'row' or 'column'.
          - If 'row', then each row must correspond to the outcomes for a particular POVM.
          - If 'column', then each column must correspond to the outcomes for a particular POVM.
        - `start` should be a tuple of non-negative integers denoting where the data starts.
          - If `'start': (i, j)`, then the data starts at row `i` and column `j`, in either row-wise or
            column-wise format as specified by `entries`.\
            Any entry in the CSV file before row `i` and column `j` is discarded.\
            Note that Python is zero-indexed, so the first row/column will be 0, second row/column will be 1, and so on.
        - Example:\
           If the data is stored in columns, but the top row is some description of the data, then we should specify:\
           `{'csv_file_path': Path to CSV file, 'entries': 'column', 'start': (1, 0)}`

-------------
Notes:
1. If `outcome_i` denotes the array of outcomes for `POVM_i = {E_1, ..., E_Ni}`, then the entries of the `outcomes_i`
   must be `0`, ..., `Ni - 1`, with `0` pointing to the POVM element `E_1`, `1` pointing to the POVM element `E_2`, and so on.\
   *Conforming to this ordering is necessary*, else spurious results can be expected. We use a zero-index following Python convention.
2. If Randomized Pauli measurement scheme (section II.E., PRA submission) was used to construct the estimator, the eigenvalues `(+1, -1)` must first
   be converted to indices `(0, 1)`, before supplying it to `compute_fidelity_estimate_risk` function.\
   `convert_Pauli_eigvals_to_indices` function can be used for this purpose to pre-process the outcomes.\
   Furthermore, for this measurement scheme, one should put all the outcomes into a *single* list and *not* into a different list for each Pauli.
3. `convert_Pauli_eigvals_to_indices` can also be used for other Pauli measurements where projection on subspace is involved.
   Note, however, that if these projectors were manually supplied in the YAML settings file, then there is no guarantee that the conversion will be correct.
4. In case path to a YAML or a CSV file is provided, the file must have the extension `.yaml` or `.csv`, respectively.
-------------

# Citation
Please cite <TODO> if you are using this code for estimating fidelity or expectation values.
