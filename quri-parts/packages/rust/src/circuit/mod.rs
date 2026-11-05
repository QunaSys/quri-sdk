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

/// Resolves an argument that can also be passed under a deprecated keyword
/// `old_name`, mirroring `quri_parts.core.utils.deprecation.deprecated_kwarg`
/// for PyO3 methods, which Python decorators cannot wrap.
pub(crate) fn resolve_deprecated_kwarg<T>(
    py: Python<'_>,
    func_name: &str,
    new_name: &str,
    new: Option<T>,
    old_name: &str,
    old: Option<T>,
) -> PyResult<T> {
    match (new, old) {
        (Some(_), Some(_)) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{func_name}() got both '{old_name}' and '{new_name}'; pass only '{new_name}'."
        ))),
        (Some(value), None) => Ok(value),
        (None, Some(value)) => {
            let message = std::ffi::CString::new(format!(
                "The '{old_name}' keyword argument is deprecated and will be removed \
                 in a future release; use '{new_name}' instead."
            ))?;
            PyErr::warn(
                py,
                &py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
                &message,
                1,
            )?;
            Ok(value)
        }
        (None, None) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{func_name}() missing 1 required positional argument: '{new_name}'"
        ))),
    }
}

/// `__repr__` must never raise (it can be invoked implicitly by logging,
/// debuggers, etc.), so fall back to a compact representation if the circuit
/// has more gates than the ASCII-art drawer's gate-index width supports, if
/// the drawer module is unavailable (e.g. quri-parts-rust used standalone),
/// or if drawing otherwise fails.
pub(crate) fn circuit_repr(
    slf: &Bound<'_, PyAny>,
    qubit_count: usize,
    gate_count: usize,
) -> PyResult<String> {
    if gate_count > 1000 {
        return Ok(compact_repr(slf, qubit_count, gate_count));
    }
    let drawn = PyModule::import(slf.py(), "quri_parts.circuit.utils.circuit_drawer")
        .and_then(|m| m.getattr("circuit_to_string"))
        .and_then(|f| f.call1((slf,)))
        .and_then(|r| r.extract::<String>());
    Ok(drawn.unwrap_or_else(|_| compact_repr(slf, qubit_count, gate_count)))
}

/// Builds the compact fallback representation, using the instance's actual
/// (possibly subclassed) type name so e.g. `QuantumCircuit` and
/// `ImmutableQuantumCircuit` aren't reported identically.
fn compact_repr(slf: &Bound<'_, PyAny>, qubit_count: usize, gate_count: usize) -> String {
    let class_name = slf
        .get_type()
        .name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "QuantumCircuit".to_string());
    format!("<{class_name} qubit_count={qubit_count} gate_count={gate_count}>")
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
