# Contribution Guidelines

We are happy that you are interested in contributing to QURI SDK!
Please read the following contribution guidelines.


## Issues

Issues are managed on [GitHub](https://github.com/QunaSys/quri-sdk/issues).
Please search existing issues before opening a new one.


## Contributor License Agreement

We ask you to sign our [Contributor License Agreement](https://cla.qunasys.com/) (CLA) upon submitting your contributions.
By signing the CLA you permit us (QunaSys) to use and redistribute your contributions as part of the project.
When you create a pull request for QURI Parts, you will be asked in a pull request comment to sign the CLA (unless you have already signed it).
You can sign the CLA by posting a comment on the pull request.
Once you sign the CLA, it will cover your future contributions submitted to QunaSys.


## Development

QURI SDK is a meta package for the packages defined and developed in the directories `quri-parts`, `quri-algo` and `quri-vm`.

We use [uv](https://docs.astral.sh/uv/) to manage dependencies and packaging.
Install the latest version and, from the repository root, run `uv sync --all-groups`
once to create a unified workspace virtualenv that contains all packages (`quri-parts`, `quri-algo`, `quri-vm`, 
and the meta package) installed as editable. uv automatically creates the project virtualenv at `.venv` in
the repository root. All subsequent development commands (tests, lint, docs, etc.) should be executed via
`uv run` in this root environment.


### Linting and testing

We use following tools for linting and testing.
Please make sure to run those tools and check if your code passes them.
All commands can be run in the project virtualenv by:

- Use `uv run`: for example `uv run black --config .black.toml .`, or
- Activate the virtualenv by `source .venv/bin/activate` and run the command.

#### Import formatting

```
uv run isort .
```
Note: when you run isort in the base directory, you need to prompts it to find and use
the config files in each subdirectory:
```
uv run isort . --resolve-all-configs
```

#### Code formatting

```
uv run black --config .black.toml .
```

Note: `--config .black.toml` keeps black's target-version detection per file
(matching CI); without it black infers target versions from `requires-python`
in pyproject.toml and may reformat files that CI accepts.

#### Document formatting

```
uv run docformatter -i -r .
```

#### Linting

```
uv run flake8
```

#### Type checking

```
uv run mypy .
```

Note: when you run mypy in a package directory (`packages/*/`), you need to specify the config file `mypy.ini`:

```
uv run mypy --config-file ../../mypy.ini .
```

#### Testing

```
uv run pytest
```

### Continuous integration (CI)

Once you create a pull request, the above linting and testing are executed on GitHub Actions.
All the checks need to be passed before merging the pull request.
