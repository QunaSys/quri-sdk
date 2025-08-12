use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::HashMap;

use crate::circuit::circuit::ImmutableQuantumCircuit;
use crate::circuit::gate::QuantumGate;
use crate::circuit::noise::noise_instruction::{
    CircuitNoiseInstruction, CircuitNoiseResolver, DepthIntervalNoise, GateIntervalNoise,
    GateNoiseInstruction, MeasurementNoise, QubitIndex,
};

#[pyclass(module = "quri_parts.rust.circuit.noise")]
pub struct CircuitNoiseInstance {
    resolvers: Vec<Box<dyn CircuitNoiseResolver + Send>>,
    index: usize,
    depths: HashMap<usize, usize>,
}

#[pymethods]
impl CircuitNoiseInstance {
    pub fn noises_for_gate(
        &mut self,
        gate: QuantumGate,
        circuit: &ImmutableQuantumCircuit,
    ) -> Vec<(Vec<QubitIndex>, GateNoiseInstruction)> {
        self.index += 1;
        self.resolvers
            .iter_mut()
            .flat_map(|noise| noise.noises_for_gate(gate.clone(), self.index - 1, circuit))
            .collect()
    }

    pub fn noises_for_depth(
        &mut self,
        qubits: Vec<usize>,
        circuit: &ImmutableQuantumCircuit,
    ) -> Vec<(Vec<QubitIndex>, GateNoiseInstruction)> {
        let depth = 1 + qubits
            .iter()
            .map(|q| self.depths.get(q).unwrap_or(&0))
            .max()
            .unwrap();
        qubits
            .iter()
            .flat_map(|q| {
                let range = (*self.depths.get(q).unwrap_or(&0)..depth).collect();
                self.depths.insert(*q, depth);
                self.resolvers
                    .iter()
                    .flat_map(|noise| noise.noises_for_depth(*q, &range, circuit))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

#[pyclass(module = "quri_parts.rust.circuit.noise")]
#[derive(Debug)]
pub struct NoiseModel {
    gate_noises: Vec<Py<GateNoiseInstruction>>,

    gate_noises_for_specified: HashMap<(String, Vec<QubitIndex>), Vec<usize>>,
    gate_noises_for_all_gates_specified_qubits: HashMap<Vec<QubitIndex>, Vec<usize>>,
    gate_noises_for_all_qubits_specified_gates: HashMap<String, Vec<usize>>,
    gate_noises_for_all: Vec<usize>,

    circuit_noises: Vec<Box<dyn CircuitNoiseInstruction + Send>>,

    custom_gate_filters: HashMap<usize, Py<PyAny>>,
}

#[pymethods]
impl NoiseModel {
    #[new]
    #[pyo3(signature = (noises=vec![]))]
    #[pyo3(
        text_signature = "(noises: Sequence[GateIntervalNoise | DepthIntervalNoise | MeasurementNoise | GateNoiseInstruction])"
    )]
    fn py_new(py: Python<'_>, noises: Vec<Bound<'_, PyAny>>) -> PyResult<Py<Self>> {
        let slf = Bound::new(
            py,
            Self {
                gate_noises: vec![],
                gate_noises_for_specified: HashMap::new(),
                gate_noises_for_all_gates_specified_qubits: HashMap::new(),
                gate_noises_for_all_qubits_specified_gates: HashMap::new(),
                gate_noises_for_all: vec![],
                circuit_noises: vec![],

                custom_gate_filters: HashMap::new(),
            },
        )?;
        NoiseModel::extend(&slf, noises)?;
        Ok(slf.unbind())
    }

    fn gate_noise_from_id(&self, noise_id: usize) -> Py<GateNoiseInstruction> {
        self.gate_noises[noise_id].clone()
    }

    pub fn gate_noise_list(&self) -> Vec<Py<GateNoiseInstruction>> {
        self.gate_noises.clone()
    }

    pub fn noises_for_gate<'py>(
        slf: &Bound<'py, Self>,
        gate: QuantumGate,
    ) -> Vec<(Bound<'py, PyTuple>, Py<GateNoiseInstruction>)> {
        let py = slf.py();
        let slf_ = slf.borrow();
        let mut pairs = vec![];

        pairs.extend(slf_.get_gate_noises_for_all(&gate, &slf_.gate_noises_for_all, py));
        pairs.extend(slf_.get_gate_noises_for_all_qubits_specified_gates(&gate, py));
        pairs.extend(slf_.get_gate_noises_for_all_gates_specified_qubits(&gate));
        pairs.extend(slf_.get_gate_noises_for_specified(&gate));

        pairs = pairs
            .into_iter()
            .filter(|(_, n)| {
                if let Some(f) = slf_.custom_gate_filters.get(&n) {
                    f.call1(py, (gate.clone(),))
                        .unwrap()
                        .extract::<bool>(py)
                        .unwrap()
                } else {
                    true
                }
            })
            .collect();

        pairs.sort_by(|a, b| a.1.cmp(&b.1));

        pairs
            .into_iter()
            .map(|(qs, nid)| (PyTuple::new_bound(py, qs), slf_.gate_noises[nid].clone()))
            .collect()
    }

    pub fn noises_for_circuit(&self) -> CircuitNoiseInstance {
        let resolvers = self
            .circuit_noises
            .iter()
            .map(|noise| noise.create_resolver())
            .collect();
        CircuitNoiseInstance {
            resolvers,
            index: 0,
            depths: HashMap::new(),
        }
    }

    fn add_gate_interval_noise(&mut self, noise: GateIntervalNoise) {
        self.add_circuit_noise_instr(Box::new(noise))
    }

    fn add_depth_interval_noise(&mut self, noise: DepthIntervalNoise) {
        self.add_circuit_noise_instr(Box::new(noise))
    }

    fn add_measurement_noise(&mut self, noise: MeasurementNoise) {
        self.add_circuit_noise_instr(Box::new(noise))
    }

    fn add_gate_noise(slf: &Bound<'_, Self>, noise: Py<GateNoiseInstruction>) -> usize {
        slf.borrow_mut().add_gate_noise_instr(noise, slf.py())
    }

    #[pyo3(signature = (noise, custom_gate_filter=None))]
    #[pyo3(
        text_signature = "(noise: GateIntervalNoise | DepthIntervalNoise | MeasurementNoise | GateNoiseInstruction, custom_gate_filter: Optional[Callable[[QuantumGate], bool]])"
    )]
    fn add_noise(
        slf: &Bound<'_, Self>,
        noise: Bound<'_, PyAny>,
        custom_gate_filter: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if let Ok(gate_noise) = noise.downcast::<GateNoiseInstruction>() {
            let nid = slf
                .borrow_mut()
                .add_gate_noise_instr(gate_noise.clone().unbind(), slf.py());
            if let Some(f) = custom_gate_filter {
                slf.borrow_mut().custom_gate_filters.insert(nid, f.unbind());
            }
        } else if let Ok(circuit_noise) = noise.downcast::<GateIntervalNoise>() {
            slf.borrow_mut()
                .add_circuit_noise_instr(Box::new(circuit_noise.get().clone()));
        } else if let Ok(circuit_noise) = noise.downcast::<DepthIntervalNoise>() {
            slf.borrow_mut()
                .add_circuit_noise_instr(Box::new(circuit_noise.get().clone()));
        } else if let Ok(circuit_noise) = noise.downcast::<MeasurementNoise>() {
            slf.borrow_mut()
                .add_circuit_noise_instr(Box::new(circuit_noise.get().clone()));
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported noise type",
            ));
        }

        Ok(())
    }

    #[pyo3(
        text_signature = "(noises: Sequence[GateIntervalNoise | DepthIntervalNoise | MeasurementNoise | GateNoiseInstruction])"
    )]
    fn extend(slf: &Bound<'_, Self>, noises: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        for noise in noises {
            NoiseModel::add_noise(slf, noise, None)?;
        }

        Ok(())
    }
}

impl NoiseModel {
    fn get_gate_noises_for_all_gates_specified_qubits(
        &self,
        gate: &QuantumGate,
    ) -> Vec<(Vec<QubitIndex>, usize)> {
        let ids = gate.get_qubits();
        let mut ret = vec![];

        ret.extend(ids.iter().flat_map(|q| {
            let qs = vec![*q];
            if let Some(noises) = self.gate_noises_for_all_gates_specified_qubits.get(&qs) {
                noises.iter().map(|noise| (qs.clone(), *noise)).collect()
            } else {
                vec![]
            }
        }));

        if ids.len() > 1 {
            let mut key = ids.clone();
            key.sort();
            if let Some(noises) = self.gate_noises_for_all_gates_specified_qubits.get(&key) {
                ret.extend(noises.iter().map(|noise| (ids.clone(), *noise)));
            }
        }

        ret
    }

    fn get_gate_noises_for_all_qubits_specified_gates(
        &self,
        gate: &QuantumGate,
        py: Python<'_>,
    ) -> Vec<(Vec<QubitIndex>, usize)> {
        self.get_gate_noises_for_all(
            gate,
            &self
                .gate_noises_for_all_qubits_specified_gates
                .get::<String>(
                    &gate
                        .clone()
                        .map_param(|p| Some(p))
                        .into_property()
                        .name
                        .into(),
                )
                .unwrap_or(&vec![])
                .clone(),
            py,
        )
    }

    fn get_gate_noises_for_all(
        &self,
        gate: &QuantumGate,
        noise_ids: &Vec<usize>,
        py: Python<'_>,
    ) -> Vec<(Vec<QubitIndex>, usize)> {
        let qs = gate.get_qubits();

        noise_ids
            .iter()
            .flat_map(|nid| {
                let noise = &self.gate_noises[*nid];
                if noise.borrow(py).qubit_count == 1 {
                    qs.iter().map(|q| (vec![*q], *nid)).collect()
                } else {
                    vec![(qs.clone(), *nid)]
                }
            })
            .collect()
    }

    fn get_gate_noises_for_specified(&self, gate: &QuantumGate) -> Vec<(Vec<QubitIndex>, usize)> {
        let ids = gate.get_qubits();
        let mut ret = vec![];

        let name = gate.clone().map_param(Some).into_property().name;

        ret.extend(ids.iter().flat_map(|q| {
            let qs = vec![*q];
            if let Some(noises) = self
                .gate_noises_for_specified
                .get(&(name.clone().into(), qs.clone()))
            {
                noises.iter().map(|noise| (qs.clone(), *noise)).collect()
            } else {
                vec![]
            }
        }));

        if ids.len() > 1 {
            let mut key = ids.clone();
            key.sort();
            if let Some(noises) = self
                .gate_noises_for_specified
                .get(&(name.clone().into(), key.clone()))
            {
                ret.extend(noises.iter().map(|noise| (ids.clone(), *noise)));
            }
        }

        ret
    }

    fn add_gate_noise_for_all_gates_specified_qubits(
        &mut self,
        noise: Py<GateNoiseInstruction>,
        nid: usize,
        py: Python<'_>,
    ) {
        let noise_ref = noise.borrow(py);
        let qcs = if noise_ref.qubit_count > 1 {
            let mut qs = noise_ref.qubit_indices.clone();
            qs.sort();
            vec![qs]
        } else {
            noise_ref.qubit_indices.iter().map(|q| vec![*q]).collect()
        };

        for qs in qcs {
            if let Some(noises) = self.gate_noises_for_all_gates_specified_qubits.get_mut(&qs) {
                noises.push(nid);
            } else {
                self.gate_noises_for_all_gates_specified_qubits
                    .insert(qs, vec![nid]);
            }
        }

        self.gate_noises.push(noise.clone());
    }

    fn add_gate_noise_for_all_qubits_specified_gates(
        &mut self,
        noise_instr: Py<GateNoiseInstruction>,
        nid: usize,
        py: Python<'_>,
    ) {
        let noise = noise_instr.borrow(py);
        for gate in noise.target_gates.iter() {
            if let Some(noises) = self
                .gate_noises_for_all_qubits_specified_gates
                .get_mut(gate)
            {
                noises.push(nid);
            } else {
                self.gate_noises_for_all_qubits_specified_gates
                    .insert(gate.clone(), vec![nid]);
            }
        }

        self.gate_noises.push(noise_instr.clone());
    }

    fn add_gate_noise_for_all(&mut self, noise_instr: Py<GateNoiseInstruction>, nid: usize) {
        self.gate_noises_for_all.push(nid);
        self.gate_noises.push(noise_instr);
    }

    fn add_gate_noise_for_specified(
        &mut self,
        noise_instr: Py<GateNoiseInstruction>,
        nid: usize,
        py: Python<'_>,
    ) {
        let noise = noise_instr.borrow(py);
        let qcs = if noise.qubit_count > 1 {
            let mut qs = noise.qubit_indices.clone();
            qs.sort();
            vec![qs]
        } else {
            noise.qubit_indices.iter().map(|q| vec![*q]).collect()
        };

        for qs in qcs {
            for gate in noise.target_gates.iter() {
                if let Some(noises) = self
                    .gate_noises_for_specified
                    .get_mut(&(gate.clone(), qs.clone()))
                {
                    noises.push(nid);
                } else {
                    self.gate_noises_for_specified
                        .insert((gate.clone(), qs.clone()), vec![nid]);
                }
            }
        }

        self.gate_noises.push(noise_instr.clone());
    }

    fn add_gate_noise_instr(
        &mut self,
        noise_instr: Py<GateNoiseInstruction>,
        py: Python<'_>,
    ) -> usize {
        let nid = self.gate_noises.len();

        let noise = noise_instr.borrow(py);

        if noise.qubit_indices.is_empty() && noise.target_gates.is_empty() {
            self.add_gate_noise_for_all(noise_instr.clone(), nid);
        } else if noise.qubit_indices.is_empty() {
            self.add_gate_noise_for_all_qubits_specified_gates(noise_instr.clone(), nid, py);
        } else if noise.target_gates.is_empty() {
            self.add_gate_noise_for_all_gates_specified_qubits(noise_instr.clone(), nid, py);
        } else {
            self.add_gate_noise_for_specified(noise_instr.clone(), nid, py);
        }

        nid
    }

    fn add_circuit_noise_instr(&mut self, noise: Box<dyn CircuitNoiseInstruction + Send>) {
        self.circuit_noises.push(noise);
    }
}
