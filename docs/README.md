# QURI SDK documentation

This documentation site is built with [Sphinx](https://www.sphinx-doc.org/) using
MyST Markdown (myst-nb) and sphinx-external-toc.

## Development

### Requirements

- Python 3.10 or 3.11 and [uv](https://docs.astral.sh/uv/)

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
$ uv sync --group doc
$ uv run python -c \
    'from juliacall import Main as jl; jl.seval("using Pkg; Pkg.add(\"PythonCall\")")'
```

Then build from this directory:

```
$ make api
```

### Rules, Conventions
- Keep all tutorial and example notebook sources under `source/docs/`; this directory is their canonical location.
- Please use Jupyter notebooks for creating or updating notebook-based documentation pages.
- Place each notebook directly in its section directory and use a descriptive, lowercase `snake_case` filename without a numeric prefix.
- Do not create a directory solely for a single notebook.
- Embed images used by a notebook directly in the `.ipynb` file instead of adding separate image files.
- Keep non-image files required to execute a notebook as external inputs and reference them with relative paths.
- Reserve `index.md` for section landing pages.
- Add notebooks directly to [`source/_toc.yml`](source/_toc.yml).
- Do not commit generated `index.md` files or notebook output assets.
- Page ordering and the site structure are defined in [`source/_toc.yml`](source/_toc.yml).
- Wrap the output cell of a Jupyter notebook in three backticks (` ``` `).
