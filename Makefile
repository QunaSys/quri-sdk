.PHONY:	develop
develop:	quri-parts/packages/rust/src quri-parts/packages/rust/Cargo.toml quri-parts/packages/rust/pyproject.toml
	uv sync --all-groups
