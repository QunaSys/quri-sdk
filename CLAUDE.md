All tools should be run in project root. run `poetry install` if error

## Linting (Formatting) Commands
For directory:
```bash
poetry run black <dir>
poetry run isort <dir>
poetry run flake8 <dir> --exclude=venv,.venv,sdist
poetry run docformatter -i --recursive <dir> --exclude venv .venv sdist
poetry run mypy <dir>
```

For single file:
```bash
poetry run black <file.py>
poetry run isort <file.py>
poetry run flake8 <file.py>
poetry run docformatter -i <file.py>
poetry run mypy <file.py>
```

## run after modify Rust code
```bash
make
cargo fmt --manifest-path quri-parts/packages/rust/Cargo.toml
```

## Testing Commands
For Python code:
```bash
poetry run pytest <dir or test files> --ignore=venv --ignore=.venv --ignore=sdist
```

For Rust code:
```bash
cargo test --manifest-path quri-parts/packages/rust/Cargo.toml
```
