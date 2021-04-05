# Minimax method for fidelity estimation

This pacakge can be used for fidelity estimation using the minimax method.

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

One can install these packages in a virtual environment.\
The file `dependencies/dependencies.yml` contains a list of packages that can be used to recereate the environment used to test the code.\
Install [Anaconda/Miniconda](https://conda.io/projects/conda/en/latest/index.html) and run `conda env create -f dependencies.yml' in the folder containing `dependencies.yml` file.
This will create an environment with the name `minimax_fidelity`. To start using this environment, run `conda activate minimax_fidelity`.

# Usage:
The code can be interactively via the Python/IPython console or through the commandline. We provide instructions for both these.

The typical workflow is as follows:
1. Specify the target state, measurement settings, and the confidence level through a YAML file. ([YAML](https://pyyaml.org/) is a human-readable markup language, which can be parsed by a computer.)
2. Create an estimator for fidelity using `handle_fidelity_estimation.py` and the YAML file. This estimator is stored in a [JSON](https://www.json.org/json-en.html) file, and can be reused in the future for the same settings.
3. Supply the outcomes to `handle_fidelity_estimation.py` along with the JSON file for the estimator to obtain the fidelity estimate and the risk.

We will postpone the details on how to specify the settings using a YAML file to [YAML settings file](#yaml) section.

## Interactive usage:
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
+----------------------|----------------------------------------+
| `yaml_filename` | Name of the YAML settings file |
| `estimator_filename` | Name of the file to which the estimator will be saved after it is constructed |
| `yaml_file_dir` | Name of the directory containing the YAML settings file. The defult value is the sub-directory `yaml_files` of the project (root) directory. This sub-directory must be created beforehand. |
| `estimator_dir` | Name of the directory to which the estimator file will be saved. The defult value is the sub-directory `estimator_files` of the project (root) directory. This sub-directory is created if it doesn't exist. |
| `print_progress` | Specify whether the progress of optimization is printed |
+----------------------|----------------------------------------+

---------
**Note**: The risk is computed along with the estimator, *before* the outcomes are supplied. This risk is printed along with the estimate when outcomes are supplied. This risk is stored in the JSON file containing the estimator.
---------

### Estimating the fidelity from outcomes:
The estimator can be constructed using the function `construct_fidelity_estimate_risk`. The syntax is as follows.

```
compute_fidelity_estimate_risk(outcomes, estimator_filename, estimator_dir = './estimator_files')
```

#### Options:
+----------------------|----------------------------------------+
| `outcomes` | The outcomes can be supplied to the function in three different ways:\
1. A list of lists/arrays, with each list/array corresponding to the outcomes of a particular POVM, listed in the same order as YAML settings file.
2. A path to a YAML file (with `.yaml` extension) that lists the outcomes corresponding to each POVM.
3. A path to a CSV file (with `.csv` extension) that lists the outcomes for each POVM in a separate row.
A detailed description of how to specify outcomes is given in section [Specifying outcomes](#outcomes) |
| `estimator_filename` | Name of the JSON file (with `.json` extension) containing the constructed estimator. |
| `estimator_dir` | Name of the directory containing the estimator file. The defult value is the sub-directory `estimator_files` of the project (root) directory. |
+----------------------|----------------------------------------+

# <a name="yaml">YAML<\a> settings file:

# Specifying <a name="outcomes">outcomes<\a>:
