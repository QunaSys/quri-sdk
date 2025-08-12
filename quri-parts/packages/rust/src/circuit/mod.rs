use pyo3::prelude::*;

pub mod circuit;
pub mod circuit_parametric;
pub mod gate;
pub mod gates;
pub mod noise;
pub mod parameter;

#[derive(Clone, Debug, PartialEq)]
pub enum MaybeUnbound {
    Bound(f64),
    Unbound(parameter::Parameter),
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "circuit")?;
    m.add_submodule(&gate::py_module(py)?)?;
    m.add_submodule(&gates::py_module(py)?)?;
    m.add_submodule(&circuit::py_module(py)?)?;
    m.add_submodule(&circuit_parametric::py_module(py)?)?;
    m.add_submodule(&parameter::py_module(py)?)?;
    m.add_submodule(&noise::py_module(py)?)?;
    Ok(m)
}
