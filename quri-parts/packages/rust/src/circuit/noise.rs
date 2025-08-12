use pyo3::prelude::*;

pub mod noise_instruction;
pub mod noise_model;

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "noise")?;
    m.add_class::<noise_model::CircuitNoiseInstance>()?;
    m.add_class::<noise_model::NoiseModel>()?;
    m.add_class::<noise_instruction::GateNoiseInstruction>()?;
    m.add_class::<noise_instruction::GateIntervalNoise>()?;
    m.add_class::<noise_instruction::DepthIntervalNoise>()?;
    m.add_class::<noise_instruction::MeasurementNoise>()?;
    Ok(m)
}
