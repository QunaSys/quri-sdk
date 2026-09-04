use pyo3::prelude::*;

pub mod circuit;
pub mod circuit_parametric;
pub mod gate;
pub mod gates;
pub mod inverse;
pub mod noise;
pub mod parameter;

#[derive(Clone, Debug, PartialEq)]
pub enum MaybeUnbound {
    Bound(f64),
    Unbound(parameter::Parameter),
}

/// `__repr__` must never raise (it can be invoked implicitly by logging,
/// debuggers, etc.), so fall back to a compact representation if the circuit
/// has more gates than the ASCII-art drawer's gate-index width supports, if
/// the drawer module is unavailable (e.g. quri-parts-rust used standalone),
/// or if drawing otherwise fails.
pub(crate) fn circuit_repr(
    slf: &Bound<'_, PyAny>,
    class_name: &str,
    qubit_count: usize,
    gate_count: usize,
) -> PyResult<String> {
    let compact = || format!("<{class_name} qubit_count={qubit_count} gate_count={gate_count}>");
    if gate_count > 1000 {
        return Ok(compact());
    }
    let drawn = PyModule::import(slf.py(), "quri_parts.circuit.utils.circuit_drawer")
        .and_then(|m| m.getattr("circuit_to_string"))
        .and_then(|f| f.call1((slf,)))
        .and_then(|r| r.extract::<String>());
    Ok(drawn.unwrap_or_else(|_| compact()))
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new(py, "circuit")?;
    m.add_submodule(&gate::py_module(py)?)?;
    m.add_submodule(&gates::py_module(py)?)?;
    m.add_submodule(&circuit::py_module(py)?)?;
    m.add_submodule(&circuit_parametric::py_module(py)?)?;
    m.add_submodule(&inverse::py_module(py)?)?;
    m.add_submodule(&parameter::py_module(py)?)?;
    m.add_submodule(&noise::py_module(py)?)?;
    Ok(m)
}
