extern crate blas_src as _; // force OpenBLAS linkage for ndarray's BLAS backend
extern crate openblas_src as _; // keep pkg-config hints active when using system OpenBLAS
use pyo3::prelude::*;

mod circuit;
mod qulacs;

use circuit::py_module as py_circuit_module;
use qulacs::py_module as py_qulacs_module;

#[pymodule]
fn quri_parts_rust(py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_submodule(&py_circuit_module(py)?)?;
    m.add_submodule(&py_qulacs_module(py)?)?;
    Ok(())
}
