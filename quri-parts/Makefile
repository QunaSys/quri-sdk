.PHONY:	develop
develop:	packages/rust/src packages/rust/Cargo.toml packages/rust/pyproject.toml
	poetry run pip install "setuptools>=64" "setuptools-rust"
	poetry run pip install "setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-manylinux_2_17_x86_64.whl ; sys_platform=='linux' and platform_machine=='x86_64'" \
	"setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-macosx_11_0_arm64.whl ; sys_platform=='darwin' and platform_machine== 'arm64'" \
	"setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-macosx_10_12_x86_64.whl ; sys_platform=='darwin' and platform_machine== 'x86_64'" \
	"setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-win_amd64.whl ; sys_platform=='win32' and platform_machine=='AMD64'" \
	"setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-manylinux_2_17_aarch64.whl ; sys_platform=='linux' and platform_machine=='aarch64'" \
	"setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-manylinux_2_17_armv7l.whl ; sys_platform=='linux' and platform_machine=='armv7l'" \
	"setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-manylinux_2_17_i686.whl ; sys_platform=='linux' and platform_machine=='i686'" \
	"setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-manylinux_2_17_ppc64le.whl ; sys_platform=='linux' and platform_machine=='ppc64le'" \
	"setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-manylinux_2_17_riscv64.whl ; sys_platform=='linux' and platform_machine=='riscv64'" \
	"setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-manylinux_2_17_s390x.whl ; sys_platform=='linux' and platform_machine=='s390x'" \
	"setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-win_arm64.whl ; sys_platform=='win32' and platform_machine=='ARM64'" \
	"setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-freebsd_14_2_release_amd64.whl ; sys_platform=='freebsd14' and platform_machine=='amd64'" \
	"setuptools-rust-bundled@https://github.com/QunaSys/setuptools-rust-bundled/releases/download/v0.1.4/setuptools_rust_bundled-0.1.4-py3-none-netbsd_10_1_amd64.whl ; sys_platform=='netbsd10' and platform_machine=='amd64'" \
	"setuptools-rust-bundled==0.1.4"
	poetry run pip install --no-build-isolation -vv -e packages/rust
