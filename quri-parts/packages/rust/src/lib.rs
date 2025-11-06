use pyo3::prelude::*;

extern crate blas_src;

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
