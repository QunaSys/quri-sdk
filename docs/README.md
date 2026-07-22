# QURI SDK documentation

This documentation site is built with [Sphinx](https://www.sphinx-doc.org/) using
MyST Markdown (myst-nb) and sphinx-external-toc.

## Development

### Requirements

- Python 3.10 or 3.11 and [Poetry](https://python-poetry.org/)

### Setting up the development environment (one-time)

```
$ make install
```

### Running the development server

```
$ make serve
```

Starts a live-reload server that rebuilds the site on save. Run `make help` for all targets.

### Build

```
$ make html
```

Generates the static site into `_build/html`, which can be served by any static host.

### API reference

Building the QURI Parts API reference is slow and uses the SDK environment at
the repository root, plus Julia and pandoc:

Install the SDK and API Reference dependencies from the repository root first:

```
$ poetry install --with dev,doc --sync
$ poetry run python -c \
    'from juliacall import Main as jl; jl.seval("using Pkg; Pkg.add(\"PythonCall\")")'
```

Then build from this directory:

```
$ make api
```

### Rules, Conventions
- Please use Jupyter notebooks for creating/updating a page for Tutorials and Examples.
- Examples are loaded directly from their `.ipynb` files; reference the notebook itself in [`source/_toc.yml`](source/_toc.yml).
- Tutorials are still authored as a pre-converted `index.md` saved alongside the notebook in the same directory.
- Page ordering and the site structure are defined in [`source/_toc.yml`](source/_toc.yml).
- Wrap the output cell of a Jupyter notebook in three backticks (` ``` `).
