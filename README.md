# QURI SDK


QURI SDK is an open source library suite for creating and executing quantum algorithms on various quantum computers and simulators. QURI SDK focuses on the followings:

- **Modularity and extensibility**: It provides small parts with which you can assemble your own algorithms. You can also use ready-made algorithms, customizing their details by replacing sub components easily.
- **Platform independence**: Once you assemble an algorithm with QURI SDK, you can execute it on various quantum computers or simulators without modifying the main algorithm code. Typically you only need to replace a few lines to switch to a different device or simulator.
- **Performance**: When dealing with a simulator, it is often the case that classical computation before and after quantum circuit simulation (such as data preparation and post processing) takes considerable time, spoiling performance of the simulator. We put an emphasis on computational performance and try to get the most out of simulators.


## Covered areas and components

QURI SDK is a meta-package that bundles the following packages, ensuring that versions are always compatible.

- Core components
  - `quri-parts`: Quantum circuit and operator library. QURI Parts contains abstractions that represent generic quantum circuits and operators, including noise, estimators, samplers, etc. It also features a wide range of transpilers and interfaces with commonly used quantum SDKs. In addition it features a quantum circuit synthesis module that allows for advanced construction of circuits for fault tolerant quantum computing (FTQC). QURI Parts is platform independent -- quantum simulation and device execution is facilitated through backends defined in various subpackages
    - Quantum circuit simulators
      - `quri-parts-qulacs`: [Qulacs](https://github.com/qulacs/qulacs)
      - `quri-parts-stim`: [Stim](https://github.com/quantumlib/Stim)
      - `quri-parts-itensor`: [ITensor](https://github.com/ITensor/ITensors.jl)
      - `quri-parts-tket`: [TKet](https://www.quantinuum.com/developers/tket)
    - Quantum platforms/SDKs
      - `quri-parts-braket`: [Amazon Braket SDK](https://github.com/aws/amazon-braket-sdk-python)
      - `quri-parts-cirq`: [Cirq](https://quantumai.google/cirq) (Only circuit and operator conversion is supported yet)
      - `quri-parts-qiskit`: [Qiskit](https://qiskit.org/)
      - `quri-parts-quantinuum`: [Quantinuum](https://www.quantinuum.com/hardware/)
      - `quri-parts-ionq`: [IONQ](https://ionq.com/docs/)
    - Intermediate representation support
      - `quri-parts-openqasm`: [OpenQASM 3.0](https://openqasm.com/)
    - Higher level packages
      - `quri-parts-qsub`: Structured circuit generation for FTQC algorithms
      - `quri-parts-algo`: NISQ algorithm components
        - Ansatz, optimizer, error mitigation etc.
    - Chemistry
      - `quri-parts-chem`: General concepts
        - Fermion-qubit mapping, electron integrals, ansatz etc.
      - `quri-parts-pyscf`: [PySCF](https://pyscf.org/)
      - Library support
        - `quri-parts-openfermion`: [OpenFermion](https://quantumai.google/openfermion)
  - `quri-algo`: Quantum algorithms and components for FTQC and Early FTQC. Contains time-evolution circuit factories and abstractions for algorithms, cost-functions, various problem definitions. It uses a modular design in which individual solvers and objective functions can be replaced or rewritten easily. Algorithms in QURI Algo are written to interface with QURI VM, although they use an abstract interface that can be used with user-defined virtual machines defined independently of QURI SDK.
  - `quri-vm`: Quantum virtual machine library written to facilitate quantum resource estimation and device aware quantum circuit simulation. The user defines a VM which generates its own sampler and estimator as well as an analyzer for quantum circuit resource analysis.


## Installation

QURI SDK requires Python 3.10 or later.

Use `pip` to install QURI Parts.
Default installation only contains components not depending specific platforms (devices/simulators) or external libraries.
You need to specify *extras* with square brackets (`[]`) to use those platforms and external libraries with QURI Parts:

```
# Default installation, no extras
pip install quri-sdk

# Use Qulacs, a quantum circuit simulator through its quri-parts interface
pip install "quri-sdk[qulacs]"

# Use Amazon Braket SDK
pip install "quri-sdk[braket]"

# Use Qulacs and OpenFermion, a quantum chemistry library for quantum computers
pip install "quri-sdk[qulacs,openfermion]"
```

Currently available extras for `quri-parts` are:

- `qulacs`
- `braket`
- `qiskit`
- `cirq`
- `openfermion`
- `stim`
- `openqasm`
- `itensor`
- `qsub`
- `quantinuum`
- `ionq`
- `pyscf`
- `tket`
- `tensornetwork`

You can also install individual components (`quri-parts-*`) directly.
In fact, `quri-parts` is a meta package, a convenience method to install those individual components.

## Documentation and tutorials

Documentation of QURI SDK is available at <https://quri-sdk.qunasys.com/>.
[Tutorials](https://quri-sdk.qunasys.com/docs/tutorials) would be a good starting point.

## Release notes

See [Releases page](https://github.com/QunaSys/quri-sdk/releases) on GitHub.

## Contribution guidelines

If you are interested in contributing to QURI Parts, please take a look at our [contribution guidelines](CONTRIBUTING.md).


## Authors

QURI SDK developed and maintained by [QunaSys Inc.](https://qunasys.com/en). All contributors can be viewed on [GitHub](https://github.com/QunaSys/quri-sdk/graphs/contributors).


## License

QURI Parts is licensed under [Apache License 2.0](https://github.com/QunaSys/quri-sdk/blob/main/quri-parts/LICENSE).
QURI Algo is licensed under an [MIT](https://github.com/QunaSys/quri-sdk/blob/main/quri-algo/LICENSE) license.
QURI VM is licensed under an [MIT](https://github.com/QunaSys/quri-sdk/blob/main/quri-vm/LICENSE) license.
