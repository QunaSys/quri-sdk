# Welcome to QURI Algo!

QURI Algo is an open source library based on [QURI Parts](https://github.com/QunaSys/quri-parts) featuring ready to use quantum algorithms. QURI Algo provides
- **Interfaces and definitions**: These are built on top of quri-parts and provide convenient abstractions for time-evolution circuits, Hadamard tests, circuit compilation, as well as a type hierarchy for abstract problems, which currently support Quantum Hamiltonians
- **Algorithms**: Quantum algorithms that can be deployed on hardware or simulated directly using estimators and samplers with or without noise
- **Compatibility**: The algorithms provided are fully compatible with QURI Parts. As such they can be easily transpiled to any architecture supported by QURI Parts and other tools made available through the QURI SDK
- **Tutorials**: The tutorials written are instructional in the algorithms used and show the logic behind them as well as showcasing our implementations

To get an overview of QURI Algo, we recommend checking out the tutorials.

## Getting started

Presently QURI Algo requires Python 3.11.1 or later. We recommend installing QURI Algo in a virtual environment using poetry or pip directly.

For the poetry installation, first make sure to upgrade your pip and virtualenv package. Then create a virtual environment, activate it and within it install poetry

```bash
$ python -m venv .venv
$ source .venv/bin/activate
(.venv)$ python -m pip install poetry
```

Then use poetry to install dependencies.

```bash
(.venv)$ python -m poetry install
```

Otherwise you can install the requirements from the requirements.txt file as

```bash
(.venv)$ pip install -r requirements.txt
```

## Documentation

Documentation to QURI Algo is available at [QURI SDK documentation site](https://quri-sdk.qunasys.com/).

## Authors

QURI Algo is developed and maintained by [QynaSys Inc](https://qunasys.com/en/). All contributors can be viewd on [github](https://github.com/QunaSys/quri-algo/graphs/contributors).

## License

QURI Algo is released under an [MIT License](https://github.com/QunaSys/quri-vm/blob/main/LICENSE).