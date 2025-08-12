# Contribution Guidelines

We are happy that you are interested in contributing to QURI Parts!
Please read the following contribution guidelines.


## Issues

Issues are managed on [GitHub](https://github.com/QunaSys/quri-parts/issues).
Please search existing issues before opening a new one.


## Contributor License Agreement

We ask you to sign our [Contributor License Agreement](https://cla.qunasys.com/) (CLA) upon submitting your contributions.
By signing the CLA you permit us (QunaSys) to use and redistribute your contributions as part of the project.
When you create a pull request for QURI Parts, you will be asked in a pull request comment to sign the CLA (unless you have already signed it).
You can sign the CLA by posting a comment on the pull request.
Once you sign the CLA, it will cover your future contributions submitted to QunaSys.


## Development

We use [Poetry](https://python-poetry.org/) to manage dependencies and packaging.
Install the latest version and run `poetry install` to create a virtualenv and install dependencies.


### Linting and testing

We use following tools for linting and testing.
Please make sure to run those tools and check if your code passes them.
All commands can be run in the Poetry virtualenv by:

- Use `poetry run`: for example `poetry run black .`, or
- Activate the virtualenv by `poetry shell` and run the command.

#### Import formatting

```
poetry run isort .
```

#### Code formatting

```
poetry run black .
```

#### Document formatting

```
poetry run docformatter -i -r .
```

#### Linting

```
poetry run flake8
```

#### Type checking

```
poetry run mypy .
```

Note: when you run mypy in a package directory (`packages/*/`), you need to specify the config file `mypy.ini`:

```
poetry run mypy --config-file ../../mypy.ini .
```

#### Testing

```
poetry run pytest
```

### Documentation

You can build documentation by:

```
cd docs
poetry run make html
# For live preview
poetry run make livehtml
```

### Continuous integration (CI)

Once you create a pull request, the above linting and testing are executed on GitHub Actions.
All the checks need to be passed before merging the pull request.
