use crate::circuit::circuit::{ImmutableQuantumCircuit, QuantumCircuit};
use crate::circuit::gate::QuantumGate;
use num_complex::Complex64;
use pyo3::prelude::*;
use quri_parts::BasicBlock;

pub fn inverse_gate(gate: &QuantumGate) -> QuantumGate {
    match gate {
        QuantumGate::S(q) => QuantumGate::Sdag(*q),
        QuantumGate::Sdag(q) => QuantumGate::S(*q),
        QuantumGate::SqrtX(q) => QuantumGate::SqrtXdag(*q),
        QuantumGate::SqrtXdag(q) => QuantumGate::SqrtX(*q),
        QuantumGate::SqrtY(q) => QuantumGate::SqrtYdag(*q),
        QuantumGate::SqrtYdag(q) => QuantumGate::SqrtY(*q),
        QuantumGate::T(q) => QuantumGate::Tdag(*q),
        QuantumGate::Tdag(q) => QuantumGate::T(*q),
        QuantumGate::RX(q, angle) => QuantumGate::RX(*q, -*angle),
        QuantumGate::RY(q, angle) => QuantumGate::RY(*q, -*angle),
        QuantumGate::RZ(q, angle) => QuantumGate::RZ(*q, -*angle),
        QuantumGate::U1(q, lmd) => QuantumGate::U1(*q, -*lmd),
        QuantumGate::U2(q, phi, lmd) => QuantumGate::U2(*q, -*phi, -*lmd),
        QuantumGate::U3(q, theta, phi, lmd) => QuantumGate::U3(*q, -*theta, -*phi, -*lmd),
        QuantumGate::PauliRotation(target_indices, pauli_ids, angle) => {
            QuantumGate::PauliRotation(target_indices.clone(), pauli_ids.clone(), -*angle)
        }
        QuantumGate::UnitaryMatrix(target_indices, unitary) => {
            let n = unitary.len();
            let mut inverse_unitary = vec![vec![Complex64::default(); n]; n];
            for (i, row) in unitary.iter().enumerate() {
                for (j, value) in row.iter().enumerate() {
                    inverse_unitary[j][i] = value.conj();
                }
            }
            QuantumGate::UnitaryMatrix(target_indices.clone(), inverse_unitary)
        }
        _ => gate.clone(),
    }
}

pub fn inverse_circuit(circuit: &ImmutableQuantumCircuit) -> ImmutableQuantumCircuit {
    let inverse_gates = circuit
        .gates
        .0
        .iter()
        .rev()
        .map(inverse_gate)
        .collect::<Vec<_>>();
    ImmutableQuantumCircuit {
        qubit_count: circuit.qubit_count,
        cbit_count: 0,
        gates: BasicBlock(inverse_gates.into()),
        depth_cache: None,
        is_immutable: false,
    }
}

#[pyfunction(
    name = "inverse_gate",
    signature = (gate),
    text_signature = "(gate: QuantumGate)",
)]
fn py_inverse_gate(gate: QuantumGate) -> QuantumGate {
    inverse_gate(&gate)
}

#[pyfunction(
    name = "inverse_circuit",
    signature = (circuit),
    text_signature = "(circuit: ImmutableQuantumCircuit)",
)]
fn py_inverse_circuit(
    circuit: &Bound<'_, ImmutableQuantumCircuit>,
) -> PyResult<Py<QuantumCircuit>> {
    let inverse = inverse_circuit(&circuit.borrow());
    Py::new(circuit.py(), (QuantumCircuit(), inverse))
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "inverse")?;
    m.add_wrapped(wrap_pyfunction!(py_inverse_gate))?;
    m.add_wrapped(wrap_pyfunction!(py_inverse_circuit))?;
    Ok(m)
}
