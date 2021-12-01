# Minimax method for fidelity estimation

This package can be used for fidelity estimation using the minimax method.

The module `handle_fidelity_estimation.py` provides a "frontend" for using the package. This code can be used to implement the minimax method from either the Python console or from a terminal.\
The code makes it easy to specify the target state and measurement settings in order to estimate the fidelity.

For a quick introduction, check the examples given in the IPython notebook `fidelity_estimation_examples.ipynb`.

# Installation and Dependencies
The code can be used "out of the box". No installation is necessary.\
We list the dependencies that must be installed for using the code.\
The version number using which the code was tested is written in brackets, next to the package.

- python (3.8.1)
- ipython (7.12.0)
- pyyaml (5.3.1)
- numpy (1.18.1)
- scipy (1.4.1)
- matplotlib (3.1.3)
- jupyter-notebook

# Usage
The code can be used interactively via the Python/IPython console or through the commandline.

The typical workflow is as follows:
1. Specify the target state, measurement settings, and the confidence level through a YAML file. ([YAML](https://pyyaml.org/) is a human-readable markup language, which can be parsed by a computer.)
2. Create an estimator for fidelity using `handle_fidelity_estimation.py` and the YAML file. This estimator is stored in a [JSON](https://www.json.org/json-en.html) file, and can be reused in the future for the same settings.
3. Supply the outcomes to `handle_fidelity_estimation.py` along with the JSON file for the estimator to obtain the fidelity estimate and the risk.

We give an outline of interactive and commandline usage. See `documentation.md` for a detailed description.

## <a name="interactive">Interactive usage:
1. Import the `handle_fidelity_estimation.py` module from a Python/IPython console.
2. Use the function `construct_fidelity_estimator` to construct a fidelity estimator from  settings specified using a YAML file.
3. Use the function `construct_fidelity_estimate_risk` to estimate the fidelity from measurement outcomes and the constructed estimator.

### Constructing the estimator:
```
construct_fidelity_estimator(yaml_filename, estimator_filename,\
                             yaml_file_dir = './yaml_files', estimator_dir = './estimator_files',\
                             print_progress = True)
```

---------

**Note**: The risk is computed along with the estimator, *before* the outcomes are supplied. This risk is printed along with the estimate when outcomes are supplied. This risk is stored in the JSON file containing the estimator.

---------

### Estimating the fidelity from outcomes:
```
compute_fidelity_estimate_risk(outcomes, estimator_filename, estimator_dir = './estimator_files')
```

## Commandline usage:
```
python handle_fidelity_estimation.py [options]
```

#### Options:
| Option         | Description                                    |
| :---:          | :---                                           |
|-y, --yaml      | Path to the YAML file that contains the settings to construct the estimator |
|-e, --estimator | Path to the JSON file that contains the estimator |
|-o, --outcomes  | Path to the YAML/CSV file containing the outcomes. See `documentation.md` for details.  |
|-q, --quiet     | If specified, the progress of optimization is not printed (optional argument, default: 0)|
|                | `--quiet` can be a number between 0 & 3, with each number corresponding to different levels of suppression of printing to stdout.|
|-h, --help      | If specified, prints the help. Other options are ignored. |

### Intended commandline usage:

| Combination of options                                | Effect                                                                                                 |
| :---:                                                 | :---                                                                                                   |
| `--yaml` and `--estimator` are supplied               | Use the YAML settings file to construct the estimator and save it in the path specified by `estimator`.|
| `--outcomes` and `--estimator` are supplied           | Use the saved estimator file and specified the outcomes file to compute the fidelity estimate.|
| `--yaml`, `--outcomes` and `--estimator` are supplied | Use the YAML settings file to construct the estimator, save it in the path specified by 'estimator',and use the outcomes file to compute the fidelity estimate.|
| `--quiet`                                             | Optional argument that can be supplied with either 0, 1, 2, 3 above that specifies whether progress is printed. |
| `--help`                                              | Prints the help. Other options are ignored. |

*Any other combination will raise an error.*

# Documentation
See [`documentation.md`](documentation.md) for a detailed description of the functionalities.\
This contains information about the arguments to be supplied to the functions and the commandline options.\
Different formats for creating the YAML file and specifying the outcomes are also given in the documentation.

# Examples
A Jupyter notebook called [`fidelity_estimation_examples.ipynb`](examples/fidelity_estimation_examples.ipynb) with some examples (using the interactive feature) can be found in the `examples` directory.\
The `examples` directory also contains the directories `yaml_files`, `estimator_files`, and `outcome_files`. These directories have files that were used for the examples given in the Jupyter notebook.

# License
We release this software under the `MIT License`. See `LICENSE` file.

# Notes
Please cite `TODO` if you are using this code for estimating fidelity or expectation values.
