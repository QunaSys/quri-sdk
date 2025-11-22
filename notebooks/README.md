# quri-sdk-notebooks
Example and tutorial notebooks used in QURI SDK documentation

## Getting started
To get a general overview of the capabilities and workflow of QURI SDK, please see these notebooks to start with

- ### **Tutorials**
    - *Quri Parts (Basics)*
        - [Quantum Gates and Circuit](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/0_circuits/0_circuits.ipynb)
        - [Quantum States](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/2_states/2_states.ipynb)
        - [Operators](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/3_operators/3_operators.ipynb)
        - [Estimators](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/4_estimators/4_estimators.ipynb)
        - [Samplers](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/6_sampling_estimation/sampling_estimation.ipynb)
        - [Sampling Backends Introduction](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/7_real_devices/0_sampling_backends/sampling_real.ipynb)
        - [Braket Sampling Backend](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/7_real_devices/1_braket/sampling_real_braket.ipynb)
        - [Qiskit Sampling Backend](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/7_real_devices/2_qiskit/sampling_real_qiskit.ipynb)
        - [Transpiling](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/8_transpiler/transpiler.ipynb)
        - [Noise simulation](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/9_noise_error/9_noise_error.ipynb)
        - [Readout Error mitigation](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/10_error_mitigation/0_readout/0_readout.ipynb)
        - [Zero noise extrapolation](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/10_error_mitigation/1_zne/1_zne.ipynb)
        - [Error mitigation by Clifford Data Regression](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/10_error_mitigation/2_cdr/2_cdr.ipynb)
        - [Data Recording](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/11_data_recording/11_data_recording.ipynb)
        
    - *Quri Parts (Advanced)*
        - [Simulators](quri_sdk_notebooks/tutorials/3_quri-parts/1_advanced/0_simulator/0_simulator.ipynb)
        - [Parametric Circuits](quri_sdk_notebooks/tutorials/3_quri-parts/1_advanced/1_parametric/0_parametric_circuit/0_parametric_circuit.ipynb)
        - [Parametric States and Parametric Estimators](quri_sdk_notebooks/tutorials/3_quri-parts/1_advanced/1_parametric/1_estimate_parametric_state/)
        - [Gradient Estimators](quri_sdk_notebooks/tutorials/3_quri-parts/1_advanced/1_parametric/2_gradient_estimators/2_gradient_estimator.ipynb)
        - [Variational Algorithms](quri_sdk_notebooks/tutorials/3_quri-parts/1_advanced/2_variational/variational.ipynb)
        - [Introduction to Quantum Chemistry](quri_sdk_notebooks/tutorials/3_quri-parts/2_quantum-chemistry/0_introduction/0_introduction.ipynb)
        - [Representing Molecular Systems](quri_sdk_notebooks/tutorials/3_quri-parts/2_quantum-chemistry/1_mo/1_molecules.ipynb)
        - [Fermion Qubit Mapping](quri_sdk_notebooks/tutorials/3_quri-parts/0_basics/5_mapping/qubit_operator_mapping.ipynb)
        - [Molecular Hamiltonian Generation](quri_sdk_notebooks/tutorials/3_quri-parts/2_quantum-chemistry/2_hamiltonian/0_hamiltonian/0_hamiltonian.ipynb)
        - [Advanced Guide on Electron integrals](quri_sdk_notebooks/tutorials/3_quri-parts/2_quantum-chemistry/2_hamiltonian/1_advanced_guide/1_advanced_guide.ipynb)
        - [Qsub - building FTQC circuits](quri_sdk_notebooks/tutorials/3_quri-parts/1_advanced/3_qsub/basics.ipynb)

    - *Quri Algo*
        - [Time evolution Circuit Factory](quri_sdk_notebooks/tutorials/1_quri-algo/0_basics/0_time_evolution/0_time_evo.ipynb)
        - [Time evolution Estimator](quri_sdk_notebooks/tutorials/1_quri-algo/0_basics/1_estimator/1_estimator.ipynb)
        
    - *Quri VM*
        - [VM Introduction](quri_sdk_notebooks/tutorials/2_quri-vm/0_vm-introduction/0_vm-introduction.ipynb)
        - [Hadamard Test](quri_sdk_notebooks/tutorials/0_general/hadamard_tutorial.ipynb)
    


- ### **Examples**
    - *Basic/Intermediate*
        - [Bell states](quri_sdk_notebooks/examples/1_quri-parts/0_Basics/0_bell/0_bell.ipynb)
        - [Sampling Tofolli Gate](quri_sdk_notebooks/examples/1_quri-parts/0_Basics/1_toffoli/1_toffoli.ipynb)
        - [Deutsch Josza Algorithm](quri_sdk_notebooks/examples/1_quri-parts/1_Intermediate/0_DJ/0._dj_algorithm.ipynb)
        - [Bernstein Vazirani Algorithm](quri_sdk_notebooks/examples/1_quri-parts/1_Intermediate/1_BV/1._bv_algorithm.ipynb)
        - [Simons Algorithm](quri_sdk_notebooks/examples/1_quri-parts/1_Intermediate/2_simons/2._simons.ipynb)
        - [Quantum Fourier Transform](quri_sdk_notebooks/examples/1_quri-parts/2_QFT_and_QPE/0_QFT/0_qft.ipynb)

    - *Advanced*
        - [Quantum Selected Configuration Interaction (QSCI)](quri_sdk_notebooks/examples/0_quri-algo-vm/0_qsci/0_qsci.ipynb)
        - [Statistical Phase Estimation (SPE)](quri_sdk_notebooks/examples/0_quri-algo-vm/1_spe/1_spe.ipynb)
        - [Quantum Assisted Quantum Compiling (QAQC)](quri_sdk_notebooks/examples/0_quri-algo-vm/2_qaqc/2_qaqc.ipynb)
        - [Surface Code](quri_sdk_notebooks/examples/0_quri-algo-vm/4_surface_code/4_surface_code.ipynb)
        - [Scalar Field Theory](quri_sdk_notebooks/examples/0_quri-algo-vm/3_scalar_field_theory/3_scalar_field_theory.ipynb)
        - [Transverse Field Ising Model](quri_sdk_notebooks/examples/0_quri-algo-vm/6_transverse_field_ising_model/6_TFIM.ipynb)
        - [Simulating Black holes with Quantum Computer](quri_sdk_notebooks/examples/0_quri-algo-vm/5_page_curve/5_page_curve.ipynb)


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
