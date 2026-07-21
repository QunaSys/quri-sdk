# Ecosystem

QURI SDK is built on a family of open-source libraries by QunaSys. This page
collects the repositories, documentation and community channels that live
outside this site.

## Core libraries

- **[QURI Parts](https://github.com/QunaSys/quri-parts)**: building blocks for
  circuits, states, operators, samplers and estimators.
  Docs: [quri-parts.qunasys.com](https://quri-parts.qunasys.com).
- **[QURI Algo](https://github.com/QunaSys/quri-sdk/tree/main/quri-algo)**: platform-independent
  algorithm definitions, including Early-FTQC algorithms.
- **[QURI VM](https://github.com/QunaSys/quri-sdk/tree/main/quri-vm)**: evaluation and simulation of
  algorithms across architectures and devices (transpilation, resource
  estimation).

## Extensions & resources

- **[QURI Parts QSCI](https://github.com/QunaSys/quri-parts-qsci)**: the QSCI
  (Quantum-Selected Configuration Interaction) algorithm built on QURI Parts.
- **[QURI SDK documentation sources](https://github.com/QunaSys/quri-sdk/tree/main/docs/source/docs)**: the source notebooks and pages behind this site.

```{note}
Not sure which library you need? See the
[QURI SDK Architecture](../concepts/overview/index.md).
```

## Community

Questions, feature ideas and bug reports are welcome on GitHub Discussions.
Sign in with a GitHub account and start or join a topic:

- [QURI SDK](https://github.com/QunaSys/quri-sdk/discussions)

## Contributing

Issues are managed on GitHub; please search existing issues before opening a new
one. Upon submitting a contribution you will be asked to sign the
[Contributor License Agreement](https://cla.qunasys.com/) (CLA) in a pull
request comment; signing it once covers your future contributions.

Development uses [Poetry](https://python-poetry.org/): run `poetry install` to
create a virtualenv with all dependencies. Before opening a pull request, make
sure linting and tests pass (they are also enforced by CI):

```bash
poetry run isort .    # import formatting
poetry run black .    # code formatting
poetry run flake8     # linting
poetry run mypy .     # type checking (in packages/*/: --config-file ../../mypy.ini)
poetry run pytest     # tests
```
