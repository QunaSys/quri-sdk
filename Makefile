.PHONY:	develop
develop:	packages/rust/src packages/rust/Cargo.toml packages/rust/pyproject.toml
	poetry run maturin develop -m packages/rust/Cargo.toml
