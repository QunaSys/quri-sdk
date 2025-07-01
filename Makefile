.PHONY:	develop
develop:	packages/rust/src packages/rust/Cargo.toml packages/rust/pyproject.toml
	poetry run pip install --no-build-isolation -vv -e packages/rust
