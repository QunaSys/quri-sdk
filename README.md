# QURI Parts


QURI Parts is an open source library suite for creating and executing quantum algorithms on various quantum computers and simulators. QURI Parts focuses on the followings:

- **Modularity and extensibility**: It provides small parts with which you can assemble your own algorithms. You can also use ready-made algorithms, customizing their details by replacing sub components easily.
- **Platform independence**: Once you assemble an algorithm with QURI Parts, you can execute it on various quantum computers or simulators without modifying the main algorithm code. Typically you only need to replace a few lines to switch to a different device or simulator.
- **Performance**: When dealing with a simulator, it is often the case that classical computation before and after quantum circuit simulation (such as data preparation and post processing) takes considerable time, spoiling performance of the simulator. We put an emphasis on computational performance and try to get the most out of simulators.


## Covered areas and components

Current QURI Parts covers the following areas, provided as individual components.
Please note that more components will be added in future.
You are also encouraged to propose or author new components as necessary.

- Core components
  - `quri-parts-circuit`: Quantum circuit
      - Gate, circuit, noise etc.
  - `quri-parts-core`: General components
      - Operator, state, estimator, sampler etc.
- Platform (device/simulator) support
  - Quantum circuit simulators
      - `quri-parts-qulacs`: [Qulacs](https://github.com/qulacs/qulacs)
      - `quri-parts-stim`: [Stim](https://github.com/quantumlib/Stim)
  - Quantum platforms/SDKs
      - `quri-parts-braket`: [Amazon Braket SDK](https://github.com/aws/amazon-braket-sdk-python)
      - `quri-parts-cirq`: [Cirq](https://quantumai.google/cirq) (Only circuit conversion is supported yet)
      - `quri-parts-qiskit`: [Qiskit](https://qiskit.org/) (Circuit conversion and execution are not supported yet)
- Intermediate representation support
  - `quri-parts-openqasm`: [OpenQASM 3.0](https://openqasm.com/)
- `quri-parts-algo`: Algorithms
  - Ansatz, optimizer, error mitigation etc.
- Chemistry
  - `quri-parts-chem`: General concepts
      - Fermion-qubit mapping etc.
  - Library support
      - `quri-parts-openfermion`: [OpenFermion](https://quantumai.google/openfermion)


## Installation

QURI Parts requires Python 3.9.8 or later.

Use `pip` to install QURI Parts.
Default installation only contains components not depending specific platforms (devices/simulators) or external libraries.
You need to specify *extras* with square brackets (`[]`) to use those platforms and external libraries with QURI Parts:

```
# Default installation, no extras
pip install quri-parts

# Use Qulacs, a quantum circuit simulator
pip install "quri-parts[qulacs]"

# Use Amazon Braket SDK
pip install "quri-parts[braket]"

# Use Qulacs and OpenFermion, a quantum chemistry library for quantum computers
pip install "quri-parts[qulacs,openfermion]"
```

Currently available extras are as follows:

- `qulacs`
- `braket`
- `qiskit`
- `cirq`
- `openfermion`
- `stim`
- `openqasm`

You can also install individual components (`quri-parts-*`) directly.
In fact, `quri-parts` is a meta package, a convenience method to install those individual components.

### Installation from local source tree

If you check out the QURI Parts repository and want to install from that local source tree, you can use `requirements-local.txt`.
In `requirements-local.txt`, optional components are commented out, so please uncomment them as necessary.

```
pip install -r requirements-local.txt
```


## Documentation and tutorials

Documentation of QURI Parts is available at <https://quri-parts.qunasys.com>.
[Tutorials](https://quri-parts.qunasys.com/tutorials.html) would be a good starting point.

## Release notes

See [Releases page](https://github.com/QunaSys/quri-parts/releases) on GitHub.


## Contribution guidelines

If you are interested in contributing to QURI Parts, please take a look at our [contribution guidelines](CONTRIBUTING.md).


## Authors

QURI Parts developed and maintained by [QunaSys Inc.](https://qunasys.com/en). All contributors can be viewed on [GitHub](https://github.com/QunaSys/quri-parts/graphs/contributors).


## License

QURI Parts is licensed under [Apache License 2.0](https://github.com/QunaSys/quri-parts/blob/main/LICENSE).
