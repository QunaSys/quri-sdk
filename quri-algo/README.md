# Welcome to QURI Algo!

QURI Algo is an open source library based on [QURI Parts](https://github.com/QunaSys/quri-parts) featuring ready to use quantum algorithms. QURI Algo provides
- **Interfaces and definitions**: These are built on top of quri-parts and provide convenient abstractions for time-evolution circuits, Hadamard tests, circuit compilation, as well as a type hierarchy for abstract problems, which currently support Quantum Hamiltonians
- **Algorithms**: Quantum algorithms that can be deployed on hardware or simulated directly using estimators and samplers with or without noise
- **Compatibility**: The algorithms provided are fully compatible with QURI Parts. As such they can be easily transpiled to any architecture supported by QURI Parts and other tools made available through the QURI SDK
- **Tutorials**: The tutorials written are instructional in the algorithms used and show the logic behind them as well as showcasing our implementations

To get an overview of QURI Algo, we recommend checking out the tutorials.

## Getting started

Presently QURI Algo requires Python 3.11.1 or later. We recommend installing QURI Algo in a virtual environment using [uv](https://docs.astral.sh/uv/) or pip directly. If you are working inside the `quri-sdk` monorepo, install dependencies once at the repository root with `uv sync --all-groups` and run all commands via `uv run` from there.

For the uv installation, first [install uv](https://docs.astral.sh/uv/getting-started/installation/). Since `quri-algo` is a member of the uv workspace, dependencies are installed from the repository root and the shared virtual environment lives at the repository root `.venv`. From the repository root, run

```bash
$ uv sync --all-groups
```

(`--all-groups` also installs the lint, typecheck, and doc tooling; drop it for just the runtime + test dependencies.)

Otherwise you can create and activate a virtual environment yourself and install the requirements from the requirements.txt file as

```bash
$ python -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install -r requirements.txt
```

## Documentation

Documentation to QURI Algo is available at [QURI SDK documentation site](https://quri-sdk.qunasys.com/).

## Authors

QURI Algo is developed and maintained by [QynaSys Inc](https://qunasys.com/en/). All contributors can be viewd on [github](https://github.com/QunaSys/quri-algo/graphs/contributors).

## License

QURI Algo is released under an [MIT License](https://github.com/QunaSys/quri-vm/blob/main/LICENSE).
