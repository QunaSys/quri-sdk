use crate::circuit::circuit::ImmutableQuantumCircuit;
use crate::circuit::noise::noise_instruction::GateNoiseInstruction;
use crate::circuit::noise::noise_model::NoiseModel;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::qulacs::convert_add_gate;

fn make_dense_matrix<'py>(
    py: Python<'py>,
    qubits: Vec<usize>,
    unitary_matrix: Vec<Vec<f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import_bound("qulacs.gate")?
        .getattr("DenseMatrix")?
        .call1((qubits, unitary_matrix))
}

fn convert_add_pauli_noise<'py>(
    py: Python<'py>,
    _qubits: &Vec<usize>,
    pauli_noise: &GateNoiseInstruction,
    qulacs_circuit: Bound<'py, PyAny>,
    fill_identity: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let module = py.import_bound("qulacs.gate")?;
    let mut probs = pauli_noise.prob_list.clone();
    let mut gates = vec![];
    for pauli in &pauli_noise.pauli_list {
        gates.push(
            module
                .getattr("Pauli")?
                .call1((pauli_noise.qubit_indices.clone(), pauli.clone()))?,
        );
    }
    if fill_identity {
        let psum: f64 = probs.iter().sum();
        if psum < 1.0 {
            gates.push(
                module
                    .getattr("Identity")?
                    .call1((pauli_noise.qubit_indices[0],))?,
            );
            probs.push(1.0 - psum);
        }
    }
    let prob_gate = py
        .import_bound("qulacs.gate")?
        .getattr("Probabilistic")?
        .call1((probs, gates))?;
    qulacs_circuit.call_method1("add_gate", (prob_gate,))?;
    Ok(qulacs_circuit)
}

fn convert_add_probabilistic_noise<'py>(
    py: Python<'py>,
    qubits: &Vec<usize>,
    prob_noise: &GateNoiseInstruction,
    qulacs_circuit: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut matrices = vec![];
    for matrix in &prob_noise.gate_matrices {
        matrices.push(make_dense_matrix(py, qubits.clone(), matrix.clone())?);
    }
    let prob_gate = py
        .import_bound("qulacs.gate")?
        .getattr("Probabilistic")?
        .call1((prob_noise.prob_list.clone(), matrices))?;
    qulacs_circuit.call_method1("add_gate", (prob_gate,))?;
    Ok(qulacs_circuit)
}

fn convert_add_kraus_noise<'py>(
    py: Python<'py>,
    qubits: &Vec<usize>,
    kraus_noise: &GateNoiseInstruction,
    qulacs_circuit: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut matrices = vec![];
    for kraus in &kraus_noise.kraus_operators {
        matrices.push(make_dense_matrix(py, qubits.clone(), kraus.clone())?);
    }
    let cptp = py
        .import_bound("qulacs.gate")?
        .getattr("CPTP")?
        .call1((matrices,))?;
    qulacs_circuit.call_method1("add_gate", (cptp,))?;
    Ok(qulacs_circuit)
}

fn convert_add_single_param_noise<'py>(
    py: Python<'py>,
    name: &str,
    qubit: usize,
    param: f64,
    qulacs_circuit: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let gate = py
        .import_bound("qulacs.gate")?
        .getattr(name)?
        .call1((qubit, param))?;
    qulacs_circuit.call_method1("add_gate", (gate,))?;
    Ok(qulacs_circuit)
}

fn convert_add_noise<'py>(
    py: Python<'py>,
    qubits: &Vec<usize>,
    noise: &GateNoiseInstruction,
    qulacs_circuit: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut circuit = qulacs_circuit;
    match noise.name.as_str() {
        name @ ("BitFlipNoise" | "DepolarizingNoise") => {
            circuit =
                convert_add_single_param_noise(py, name, qubits[0], noise.params[0], circuit)?;
        }
        "PhaseFlipNoise" => {
            circuit = convert_add_single_param_noise(
                py,
                "DephasingNoise",
                qubits[0],
                noise.params[0],
                circuit,
            )?;
        }
        "BitPhaseFlipNoise" => {
            circuit = convert_add_single_param_noise(
                py,
                "IndependentXZNoise",
                qubits[0],
                noise.params[0],
                circuit,
            )?;
        }
        "PauliNoise" | "GeneralDepolarizingNoise" => {
            circuit = convert_add_pauli_noise(py, qubits, noise, circuit, false)?;
        }
        "ProbabilisticNoise" => {
            circuit = convert_add_probabilistic_noise(py, qubits, noise, circuit)?;
        }
        "KrausNoise"
        | "ResetNoise"
        | "PhaseDampingNoise"
        | "AmplitudeDampingNoise"
        | "PhaseAmplitudeDampingNoise"
        | "ThermalRelaxationNoise" => {
            circuit = convert_add_kraus_noise(py, qubits, noise, circuit)?;
        }
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported noise: {}",
                other
            )))
        }
    }
    Ok(circuit)
}

#[pyfunction]
pub fn convert_circuit_with_noise_model<'py>(
    py: Python<'py>,
    circuit: Bound<'py, ImmutableQuantumCircuit>,
    noise_model_instance: Bound<'py, NoiseModel>,
) -> PyResult<Bound<'py, PyAny>> {
    let circuit = circuit.borrow();
    let mut qulacs_circuit = py
        .import_bound("qulacs")?
        .getattr("QuantumCircuit")?
        .call1((circuit.qubit_count,))?;

    // make circuit noise instance & gate noise dict
    let noise_model = noise_model_instance.borrow();
    let mut circuit_noise = noise_model.noises_for_circuit();

    for gate in &circuit.gates.0 {
        let gate_qubits = gate.get_qubits();

        // circuit noises for depth
        for (qubits, noise) in circuit_noise.noises_for_depth(gate_qubits, &circuit) {
            qulacs_circuit = convert_add_noise(py, &qubits, &noise, qulacs_circuit)?;
        }

        qulacs_circuit = convert_add_gate(py, gate, qulacs_circuit)?;

        // gate noises
        for (qubits, noise) in NoiseModel::noises_for_gate(&noise_model_instance, gate.clone()) {
            qulacs_circuit =
                convert_add_noise(py, &qubits.extract()?, noise.get(), qulacs_circuit)?;
        }

        // circuit noises for gate
        for (qubits, noise) in circuit_noise.noises_for_gate(gate.clone(), &circuit) {
            qulacs_circuit = convert_add_noise(py, &qubits, &noise, qulacs_circuit)?;
        }
    }

    // circuit noises for depth
    let all_qubits = (0..circuit.qubit_count).collect();
    for (qubits, noise) in circuit_noise.noises_for_depth(all_qubits, &circuit) {
        qulacs_circuit = convert_add_noise(py, &qubits, &noise, qulacs_circuit)?;
    }

    Ok(qulacs_circuit.as_any().clone())
}
