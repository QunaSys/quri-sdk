# quri-sdk-notebooks
Example and tutorial notebooks used in QURI SDK documentation

## Getting started
To get a general overview of the capabilities and workflow of QURI SDK, please see these notebooks to start with

- `quri_sdk_notebooks/tutorials/0_general/hadamard_tutorial.ipynb`
- `quri_sdk_notebooks/tutorials/2_quri-vm/0_vm-introduction/0_vm-introduction.ipynb`
- `quri_sdk_notebooks/tutorials/1_quri-algo/0_basics/1_estimator/1_estimator.ipynb`
- `quri_sdk_notebooks/tutorials/1_quri-algo/0_basics/0_time_evolution/0_time_evo.ipynb`
- `quri_sdk_notebooks/examples/0_quri-algo-vm/0_qsci/0_qsci.ipynb`
- `quri_sdk_notebooks/examples/0_quri-algo-vm/1_spe/1_spe.ipynb`
- `quri_sdk_notebooks/examples/0_quri-algo-vm/2_qaqc/2_qaqc.ipynb`


## Installation instructions

The notebooks here use `quri-algo`, `quri-parts`, `quri-vm` and `quri-parts-qsci`. Make sure you have the latest versions of them installed.

The easiest way is to use a virtual environment with Python 3.10 or later. Then install the required packages using

```
pip install -r requirements.txt
```

Another way is to use [poetry](https://python-poetry.org/docs/#installing-with-pipx).

```
poetry install
```

Then run the notebooks in your virtual environment. You can run jupyter using

```
poetry run jupyter notebook
```

and then open the jupyter server in your browser.

## Contributing

We require that all contributed notebooks can run without throwing any error and that no metadata is included. You can clean your notebook of metadata by running the command

```
make clean-nb
```

Before opening a pull request make sure that your notebook can run in its entirety without throwing any errors. You can do so using

```
make execute-in-place
```

Beware that this executes any notebooks you have added or modified in place and overwrites output cells. If you only want to check them you can run

```
poetry run jupyter execute quri_sdk_notebooks/path/to/notebook.ipynb
```

At times we may stop maintaining certain notebooks. These will be excluded from CI and we do not guarantee that they can still run. The file `.exclude` keeps track of these.
