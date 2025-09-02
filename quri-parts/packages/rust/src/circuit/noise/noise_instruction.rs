use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt;

use crate::circuit::circuit::ImmutableQuantumCircuit;
use crate::circuit::gate::QuantumGate;

pub type QubitIndex = usize;
pub type QubitNoisePair = (Vec<QubitIndex>, GateNoiseInstruction);

pub trait CircuitNoiseResolver {
    fn noises_for_gate(
        &mut self,
        _gate: QuantumGate,
        _index: usize,
        _circuit: &ImmutableQuantumCircuit,
    ) -> Vec<QubitNoisePair> {
        vec![]
    }

    fn noises_for_depth(
        &self,
        _qubit: usize,
        _depths: &Vec<usize>,
        _circuit: &ImmutableQuantumCircuit,
    ) -> Vec<QubitNoisePair> {
        vec![]
    }
}

pub trait CircuitNoiseInstruction: fmt::Debug {
    fn create_resolver(&self) -> Box<dyn CircuitNoiseResolver + Send>;
}

#[pyclass(eq, frozen, subclass, module = "quri_parts.rust.circuit.noise")]
#[derive(Debug, Clone, PartialEq)]
pub struct GateNoiseInstruction {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub qubit_count: usize,
    #[pyo3(get)]
    pub params: Vec<f64>,
    #[pyo3(get)]
    pub qubit_indices: Vec<usize>,
    #[pyo3(get)]
    pub target_gates: Vec<String>,
    #[pyo3(get)]
    pub pauli_list: Vec<Vec<usize>>,
    #[pyo3(get)]
    pub prob_list: Vec<f64>,
    #[pyo3(get)]
    pub kraus_operators: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub gate_matrices: Vec<Vec<Vec<f64>>>,
}

#[pymethods]
impl GateNoiseInstruction {
    #[new]
    #[pyo3(signature = (name, qubit_count, params, qubit_indices, target_gates, pauli_list=vec![], prob_list=vec![], kraus_operators=vec![], gate_matrices=vec![]))]
    pub fn new(
        name: String,
        qubit_count: usize,
        params: Vec<f64>,
        qubit_indices: Vec<usize>,
        target_gates: Vec<String>,
        pauli_list: Vec<Vec<usize>>,
        prob_list: Vec<f64>,
        kraus_operators: Vec<Vec<Vec<f64>>>,
        gate_matrices: Vec<Vec<Vec<f64>>>,
    ) -> Self {
        Self {
            name,
            qubit_count,
            params,
            qubit_indices,
            target_gates,
            pauli_list,
            prob_list,
            kraus_operators,
            gate_matrices,
        }
    }
}

#[pyclass(eq, frozen, module = "quri_parts.rust.circuit.noise")]
#[derive(Debug, Clone, PartialEq)]
pub struct GateIntervalNoise {
    noises: Vec<GateNoiseInstruction>,
    interval: usize,
}

#[pymethods]
impl GateIntervalNoise {
    #[new]
    pub fn new(single_qubit_noises: Vec<GateNoiseInstruction>, gate_interval: usize) -> Self {
        Self {
            noises: single_qubit_noises,
            interval: gate_interval,
        }
    }

    fn name(&self) -> String {
        "GateIntervalNoise".to_owned()
    }
}

struct GateIntervalNoiseResolver {
    noises: Vec<GateNoiseInstruction>,
    interval: usize,
    qubit_indicators: HashMap<usize, usize>,
}

impl CircuitNoiseResolver for GateIntervalNoiseResolver {
    fn noises_for_gate(
        &mut self,
        gate: QuantumGate,
        _: usize,
        _: &ImmutableQuantumCircuit,
    ) -> Vec<QubitNoisePair> {
        let mut ret = vec![];
        for q in gate.get_qubits() {
            let qi = self.qubit_indicators.entry(q).or_insert(0);
            *qi += 1;
            if *qi >= self.interval {
                self.qubit_indicators.insert(q, 0);
                ret.extend(self.noises.iter().map(|n| (vec![q], n.clone())));
            }
        }
        ret
    }
}

impl CircuitNoiseInstruction for GateIntervalNoise {
    fn create_resolver(&self) -> Box<dyn CircuitNoiseResolver + Send> {
        Box::new(GateIntervalNoiseResolver {
            noises: self.noises.clone(),
            interval: self.interval,
            qubit_indicators: HashMap::new(),
        })
    }
}

#[pyclass(eq, frozen, module = "quri_parts.circuit.circuit.noise")]
#[derive(Debug, Clone, PartialEq)]
pub struct DepthIntervalNoise {
    noises: Vec<GateNoiseInstruction>,
    interval: usize,
}

#[pymethods]
impl DepthIntervalNoise {
    #[new]
    pub fn new(single_qubit_noises: Vec<GateNoiseInstruction>, depth_interval: usize) -> Self {
        Self {
            noises: single_qubit_noises,
            interval: depth_interval,
        }
    }

    fn name(&self) -> String {
        "DepthIntervalNoise".to_owned()
    }
}

struct DepthIntervalNoiseResolver {
    noises: Vec<GateNoiseInstruction>,
    interval: usize,
}

impl CircuitNoiseResolver for DepthIntervalNoiseResolver {
    fn noises_for_depth(
        &self,
        qubit: usize,
        depths: &Vec<usize>,
        _: &ImmutableQuantumCircuit,
    ) -> Vec<QubitNoisePair> {
        let mut ret = vec![];
        let pairs = self.noises.iter().map(|n| (vec![qubit], n.clone()));
        for d in depths {
            if *d > 0 && d % self.interval == 0 {
                ret.extend(pairs.clone());
            }
        }
        ret
    }
}

impl CircuitNoiseInstruction for DepthIntervalNoise {
    fn create_resolver(&self) -> Box<dyn CircuitNoiseResolver + Send> {
        Box::new(DepthIntervalNoiseResolver {
            noises: self.noises.clone(),
            interval: self.interval,
        })
    }
}

#[pyclass(eq, frozen, module = "quri_parts.rust.circuit.noise")]
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementNoise {
    id: usize,
    noises: Vec<GateNoiseInstruction>,
    qubit_indices: Vec<usize>,
}

struct MeasurementNoiseResolver {
    noises: Vec<GateNoiseInstruction>,
    qubit_indices: Vec<usize>,
}

impl CircuitNoiseResolver for MeasurementNoiseResolver {
    fn noises_for_depth(
        &self,
        qubit: usize,
        depths: &Vec<usize>,
        circuit: &ImmutableQuantumCircuit,
    ) -> Vec<QubitNoisePair> {
        if depths.contains(&circuit.depth()) {
            if self.qubit_indices.is_empty() || self.qubit_indices.contains(&qubit) {
                return self
                    .noises
                    .iter()
                    .map(|n| (vec![qubit], n.clone()))
                    .collect();
            }
        }
        vec![]
    }
}

#[pymethods]
impl MeasurementNoise {
    #[new]
    #[pyo3(signature = (single_qubit_noises, qubit_indices=vec![]))]
    pub fn new(single_qubit_noises: Vec<GateNoiseInstruction>, qubit_indices: Vec<usize>) -> Self {
        Self {
            id: 0,
            noises: single_qubit_noises,
            qubit_indices: qubit_indices,
        }
    }

    fn name(&self) -> String {
        "MeasurementNoise".to_owned()
    }
}

impl CircuitNoiseInstruction for MeasurementNoise {
    fn create_resolver(&self) -> Box<dyn CircuitNoiseResolver + Send> {
        Box::new(MeasurementNoiseResolver {
            noises: self.noises.clone(),
            qubit_indices: vec![],
        })
    }
}
