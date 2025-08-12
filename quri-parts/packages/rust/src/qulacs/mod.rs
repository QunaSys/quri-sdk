use crate::circuit::circuit::ImmutableQuantumCircuit;
use crate::circuit::gate::QuantumGate;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

mod noise;

pub fn convert_add_gate<'py>(
    py: Python<'py>,
    gate: &QuantumGate,
    qulacs_circuit: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match gate {
        QuantumGate::Identity(q1) => {
            let identity_gate = py
                .import_bound("qulacs.gate")?
                .getattr("Identity")?
                .call1((*q1,))?;
            qulacs_circuit.call_method1("add_gate", (identity_gate,))?;
        }
        QuantumGate::X(q1) => {
            qulacs_circuit.call_method1("add_X_gate", (*q1,))?;
        }
        QuantumGate::Y(q1) => {
            qulacs_circuit.call_method1("add_Y_gate", (*q1,))?;
        }
        QuantumGate::Z(q1) => {
            qulacs_circuit.call_method1("add_Z_gate", (*q1,))?;
        }
        QuantumGate::H(q1) => {
            qulacs_circuit.call_method1("add_H_gate", (*q1,))?;
        }
        QuantumGate::S(q1) => {
            qulacs_circuit.call_method1("add_S_gate", (*q1,))?;
        }
        QuantumGate::Sdag(q1) => {
            qulacs_circuit.call_method1("add_Sdag_gate", (*q1,))?;
        }
        QuantumGate::SqrtX(q1) => {
            qulacs_circuit.call_method1("add_sqrtX_gate", (*q1,))?;
        }
        QuantumGate::SqrtXdag(q1) => {
            qulacs_circuit.call_method1("add_sqrtXdag_gate", (*q1,))?;
        }
        QuantumGate::SqrtY(q1) => {
            qulacs_circuit.call_method1("add_sqrtY_gate", (*q1,))?;
        }
        QuantumGate::SqrtYdag(q1) => {
            qulacs_circuit.call_method1("add_sqrtYdag_gate", (*q1,))?;
        }
        QuantumGate::T(q1) => {
            qulacs_circuit.call_method1("add_T_gate", (*q1,))?;
        }
        QuantumGate::Tdag(q1) => {
            qulacs_circuit.call_method1("add_Tdag_gate", (*q1,))?;
        }
        QuantumGate::RX(q1, p1) => {
            qulacs_circuit.call_method1("add_RX_gate", (*q1, -*p1))?;
        }
        QuantumGate::RY(q1, p1) => {
            qulacs_circuit.call_method1("add_RY_gate", (*q1, -*p1))?;
        }
        QuantumGate::RZ(q1, p1) => {
            qulacs_circuit.call_method1("add_RZ_gate", (*q1, -*p1))?;
        }
        QuantumGate::U1(q1, p1) => {
            qulacs_circuit.call_method1("add_U1_gate", (*q1, *p1))?;
        }
        QuantumGate::U2(q1, p1, p2) => {
            qulacs_circuit.call_method1("add_U2_gate", (*q1, *p1, *p2))?;
        }
        QuantumGate::U3(q1, p1, p2, p3) => {
            qulacs_circuit.call_method1("add_U3_gate", (*q1, *p1, *p2, *p3))?;
        }
        QuantumGate::CNOT(q1, q2) => {
            qulacs_circuit.call_method1("add_CNOT_gate", (*q1, *q2))?;
        }
        QuantumGate::CZ(q1, q2) => {
            qulacs_circuit.call_method1("add_CZ_gate", (*q1, *q2))?;
        }
        QuantumGate::SWAP(q1, q2) => {
            qulacs_circuit.call_method1("add_SWAP_gate", (*q1, *q2))?;
        }
        QuantumGate::TOFFOLI(q1, q2, q3) => {
            let toffoli_gate = py
                .import_bound("qulacs.gate")?
                .getattr("TOFFOLI")?
                .call1((*q1, *q2, *q3))?;
            qulacs_circuit.call_method1("add_gate", (toffoli_gate,))?;
        }
        QuantumGate::UnitaryMatrix(qs, mat) => {
            qulacs_circuit.call_method1(
                "add_dense_matrix_gate",
                (
                    Vec::from(qs.clone()),
                    mat.iter()
                        .map(|v| v.clone().into())
                        .collect::<Vec<Vec<_>>>(),
                ),
            )?;
        }
        QuantumGate::Pauli(qs, pauli_ids) => {
            qulacs_circuit.call_method1(
                "add_multi_Pauli_gate",
                (Vec::from(qs.clone()), Vec::from(pauli_ids.clone())),
            )?;
        }
        QuantumGate::PauliRotation(qs, pauli_ids, angle) => {
            qulacs_circuit.call_method1(
                "add_multi_Pauli_rotation_gate",
                (Vec::from(qs.clone()), Vec::from(pauli_ids.clone()), -*angle),
            )?;
        }
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "{} is not supported",
                &other.clone().map_param(Some).into_property().name
            )))
        }
    }

    Ok(qulacs_circuit)
}

#[pyfunction]
fn convert_circuit<'py>(
    py: Python<'py>,
    circuit: Bound<'py, ImmutableQuantumCircuit>,
) -> PyResult<Bound<'py, PyAny>> {
    let circuit = circuit.borrow();
    let mut qulacs_circuit = py
        .import_bound("qulacs")?
        .getattr("QuantumCircuit")?
        .call1((circuit.qubit_count,))?;
    for gate in &circuit.gates.0 {
        qulacs_circuit = convert_add_gate(py, gate, qulacs_circuit)?;
    }
    Ok(qulacs_circuit.as_any().clone())
}

pub fn py_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new_bound(py, "qulacs")?;
    m.add_function(wrap_pyfunction!(convert_circuit, &m)?)?;
    m.add_function(wrap_pyfunction!(
        noise::convert_circuit_with_noise_model,
        &m
    )?)?;
    Ok(m)
}
