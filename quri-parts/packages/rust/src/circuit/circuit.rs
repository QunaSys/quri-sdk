use crate::circuit::gate::QuantumGate;
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::{PySequence, PyString, PyTuple};
use quri_parts::BasicBlock;
use std::collections::HashMap;

const PICKLE_STUB_ARG: &'static str = "__QURI_PARTS_STUB_ARG_FOR_UNPICKLING";

#[pyclass(subclass, eq, module = "quri_parts.rust.circuit.circuit")]
#[derive(Clone)]
#[repr(C)]
pub struct ImmutableQuantumCircuit {
    #[pyo3(get)]
    pub qubit_count: usize,
    #[pyo3(get)]
    pub cbit_count: usize,
    pub gates: BasicBlock<QuantumGate<f64>>,
    pub depth_cache: Option<usize>,
    pub is_immutable: bool,
}

impl PartialEq for ImmutableQuantumCircuit {
    fn eq(&self, other: &Self) -> bool {
        self.qubit_count == other.qubit_count
            && self.cbit_count == other.cbit_count
            && &self.gates.0.len() == &other.gates.0.len()
            && self.gates.0.iter().zip(&other.gates.0).all(|(l, r)| {
                l.clone().map_param(Some).into_property()
                    == r.clone().map_param(Some).into_property()
            })
    }
}

impl ImmutableQuantumCircuit {
    pub fn depth(&self) -> usize {
        let mut ds = HashMap::<usize, usize>::new();
        for gate in self.gates.0.iter() {
            let qubits = gate.get_qubits();
            let d = 1 + qubits
                .iter()
                .map(|q| ds.get(&q).unwrap_or(&0))
                .max()
                .unwrap();
            qubits.iter().for_each(|q| {
                ds.insert(*q, d);
            });
        }
        ds.into_values().max().unwrap_or(0)
    }
}

#[pymethods]
impl ImmutableQuantumCircuit {
    #[new]
    #[pyo3(signature = (*args))]
    #[pyo3(text_signature = "(circuit: ImmutableQuantumCircuit)")]
    fn py_new<'py>(args: &Bound<'py, PyAny>) -> PyResult<Py<Self>> {
        if let Ok((stub, qubit_count, cbit_count, gates)) =
            args.extract::<(Bound<'py, PyString>, usize, usize, Vec<QuantumGate>)>()
        {
            // Called from pickle.load()
            if stub.to_str()? == PICKLE_STUB_ARG {
                return Py::new(
                    args.py(),
                    Self {
                        qubit_count,
                        cbit_count,
                        gates: BasicBlock(gates.into()),
                        depth_cache: None,
                        is_immutable: true,
                    },
                );
            }
        }
        let immut = args.extract::<(Py<ImmutableQuantumCircuit>,)>()?.0;
        {
            let mut borrowed = immut.borrow_mut(args.py());
            borrowed.is_immutable = true;
        }
        Ok(immut)
    }

    #[pyo3(name = "__reduce__")]
    fn py_reduce(
        slf: &Bound<'_, Self>,
    ) -> PyResult<(PyObject, (Py<PyString>, usize, usize, Vec<QuantumGate>))> {
        let borrowed = slf.borrow();
        Ok((
            slf.getattr("__class__")?.unbind(),
            (
                PyString::new_bound(slf.py(), PICKLE_STUB_ARG).unbind(),
                borrowed.qubit_count,
                borrowed.cbit_count,
                borrowed.gates.0.clone().into(),
            ),
        ))
    }

    #[getter]
    fn get_gates<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        Ok(PyTuple::new_bound(
            slf.py(),
            slf.gates.0.iter().map(|g| g.clone().into_py(slf.py())),
        ))
    }

    #[getter]
    fn get_depth(mut slf: PyRefMut<'_, Self>) -> usize {
        if let Some(depth) = slf.depth_cache {
            depth
        } else {
            let d = slf.depth();
            slf.depth_cache = Some(d);
            d
        }
    }

    #[pyo3(text_signature = "(gates: ImmutableQuantumCircuit | Sequence[QuantumGate])")]
    fn combine(slf: &Bound<'_, Self>, gates: Bound<'_, PyAny>) -> PyResult<Py<QuantumCircuit>> {
        let ret = Self::get_mutable_copy(slf)?;
        QuantumCircuit::extend(ret.bind(slf.py()), gates)?;
        Ok(ret)
    }

    #[pyo3(name = "__add__")]
    fn py_add(slf: &Bound<'_, Self>, gates: Bound<'_, PyAny>) -> PyObject {
        Self::combine(slf, gates)
            .map(|c| c.into_py(slf.py()))
            .unwrap_or(slf.py().NotImplemented())
    }

    fn freeze(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        if slf.borrow().is_immutable {
            Ok(slf)
        } else {
            let mut cloned = slf.borrow().clone();
            cloned.is_immutable = true;
            Bound::new(slf.py(), cloned)
        }
    }

    fn get_mutable_copy(slf: &Bound<'_, Self>) -> PyResult<Py<QuantumCircuit>> {
        Py::new(slf.py(), (QuantumCircuit(), slf.borrow().clone()))
    }

    fn sample<'py>(slf: &Bound<'py, Self>, shot_count: i32) -> PyResult<Bound<'py, PyAny>> {
        let sampling =
            PyModule::import_bound(slf.py(), "quri_parts.core.sampling.default_sampler")?;
        let sampling_counts = sampling
            .getattr("DEFAULT_SAMPLER")?
            .call1((slf, shot_count));
        sampling_counts
    }

    fn draw<'py>(slf: &Bound<'py, Self>) -> Result<(), PyErr> {
        let circuit_drawer =
            PyModule::import_bound(slf.py(), "quri_parts.circuit.utils.circuit_drawer")?;
        circuit_drawer.getattr("draw_circuit")?.call1((slf,))?;

        Ok(())
    }
}

#[pyclass(
    extends=ImmutableQuantumCircuit,
    subclass,
    module = "quri_parts.rust.circuit.circuit"
)]
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumCircuit();

#[pymethods]
impl QuantumCircuit {
    #[new]
    #[pyo3(signature = (qubit_count, cbit_count=None, gates=Vec::new()))]
    fn py_new(
        qubit_count: usize,
        cbit_count: Option<Bound<'_, PyAny>>,
        gates: Vec<QuantumGate>,
    ) -> PyResult<(Self, ImmutableQuantumCircuit)> {
        let (cbit_count, gates) = if let Some(cbit_count) = cbit_count.as_ref() {
            if let Ok(seq) = cbit_count.downcast::<PySequence>() {
                cbit_count
                    .py()
                    .run_bound(
                        r#"
import warnings
warnings.warn(
    "QuantumCircuit initialization takes cbit_count before gates.",
    DeprecationWarning,
)
                        "#,
                        None,
                        None,
                    )
                    .unwrap();
                (
                    0,
                    (0..(seq.len()?))
                        .map(|i| FromPyObject::extract_bound(&seq.get_item(i)?))
                        .collect::<PyResult<Vec<_>>>()?,
                )
            } else {
                (FromPyObject::extract_bound(&cbit_count)?, gates)
            }
        } else {
            (0, gates)
        };

        let base = ImmutableQuantumCircuit {
            qubit_count,
            cbit_count,
            gates: BasicBlock(gates.into()),
            depth_cache: None,
            is_immutable: false,
        };
        Ok((QuantumCircuit(), base))
    }

    #[pyo3(name = "__reduce__")]
    fn py_reduce(slf: &Bound<'_, Self>) -> PyResult<(PyObject, (usize, usize, Vec<QuantumGate>))> {
        let borrowed = slf.borrow();
        let sup = borrowed.as_super();
        Ok((
            slf.getattr("__class__")?.unbind(),
            (sup.qubit_count, sup.cbit_count, sup.gates.0.clone().into()),
        ))
    }

    #[pyo3(signature = (gate, gate_index = None))]
    #[pyo3(text_signature = "(gate: QuantumGate, gate_index: Optional[int])")]
    fn add_gate(
        mut slf: PyRefMut<'_, Self>,
        gate: QuantumGate,
        gate_index: Option<usize>,
    ) -> PyResult<()> {
        slf.as_super().depth_cache.take();
        if gate
            .get_qubits()
            .iter()
            .max()
            .map(|n| n >= &slf.as_super().qubit_count)
            .unwrap_or(false)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "The indices of the gate applied must be smaller than qubit_count",
            ));
        }
        if gate
            .get_cbits()
            .iter()
            .max()
            .map(|n| n >= &slf.as_super().cbit_count)
            .unwrap_or(false)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "The classical indices of the gate applied must be smaller than cbit_count",
            ));
        }
        let gates = &mut slf.as_super().gates.0;
        if let Some(gate_index) = gate_index {
            if gate_index <= gates.len() {
                gates.insert(gate_index, gate.clone());
                Ok(())
            } else {
                Err(pyo3::exceptions::PyIndexError::new_err(""))
            }
        } else {
            gates.push(gate.clone());
            Ok(())
        }
    }

    #[pyo3(text_signature = "(gates: ImmutableQuantumCircuit | Sequence[QuantumGate])")]
    fn extend(slf: &Bound<'_, Self>, gates: Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(other) = gates.downcast::<ImmutableQuantumCircuit>() {
            for gate in other.borrow().gates.0.iter() {
                Self::add_gate(slf.borrow_mut(), gate.clone(), None)?;
            }
            Ok(())
        } else if let Ok(other) = gates.downcast::<PySequence>() {
            for i in 0..other.len()? {
                if let Ok(gate) = QuantumGate::extract_bound(&other.get_item(i)?) {
                    Self::add_gate(slf.borrow_mut(), gate, None)?;
                }
            }
            Ok(())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Bad argument `gates`",
            ))
        }
    }

    #[pyo3(name = "__iadd__")]
    fn py_iadd(slf: &Bound<'_, Self>, gates: Bound<'_, PyAny>) -> PyResult<()> {
        Self::extend(slf, gates)
            .map_err(|_| pyo3::exceptions::PyNotImplementedError::new_err("Not implemented"))
    }

    #[allow(non_snake_case)]
    fn add_Identity_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::Identity(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_X_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::X(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_Y_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::Y(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_Z_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::Z(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_H_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::H(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_S_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::S(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_Sdag_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::Sdag(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_SqrtX_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::SqrtX(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_SqrtXdag_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::SqrtXdag(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_SqrtY_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::SqrtY(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_SqrtYdag_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::SqrtYdag(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_T_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::T(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_Tdag_gate(slf: PyRefMut<'_, Self>, qubit_index: usize) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::Tdag(qubit_index), None)
    }

    #[allow(non_snake_case)]
    fn add_U1_gate(slf: PyRefMut<'_, Self>, qubit_index: usize, lmd: f64) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::U1(qubit_index, lmd), None)
    }

    #[allow(non_snake_case)]
    fn add_U2_gate(
        slf: PyRefMut<'_, Self>,
        qubit_index: usize,
        phi: f64,
        lmd: f64,
    ) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::U2(qubit_index, phi, lmd), None)
    }

    #[allow(non_snake_case)]
    fn add_U3_gate(
        slf: PyRefMut<'_, Self>,
        qubit_index: usize,
        theta: f64,
        phi: f64,
        lmd: f64,
    ) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::U3(qubit_index, theta, phi, lmd), None)
    }

    #[allow(non_snake_case)]
    fn add_RX_gate(slf: PyRefMut<'_, Self>, qubit_index: usize, angle: f64) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::RX(qubit_index, angle), None)
    }

    #[allow(non_snake_case)]
    fn add_RY_gate(slf: PyRefMut<'_, Self>, qubit_index: usize, angle: f64) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::RY(qubit_index, angle), None)
    }

    #[allow(non_snake_case)]
    fn add_RZ_gate(slf: PyRefMut<'_, Self>, qubit_index: usize, angle: f64) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::RZ(qubit_index, angle), None)
    }

    #[allow(non_snake_case)]
    fn add_CNOT_gate(
        slf: PyRefMut<'_, Self>,
        control_index: usize,
        target_index: usize,
    ) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::CNOT(control_index, target_index), None)
    }

    #[allow(non_snake_case)]
    fn add_CZ_gate(
        slf: PyRefMut<'_, Self>,
        control_index: usize,
        target_index: usize,
    ) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::CZ(control_index, target_index), None)
    }

    #[allow(non_snake_case)]
    fn add_SWAP_gate(
        slf: PyRefMut<'_, Self>,
        target_index1: usize,
        target_index2: usize,
    ) -> PyResult<()> {
        Self::add_gate(slf, QuantumGate::SWAP(target_index1, target_index2), None)
    }

    #[allow(non_snake_case)]
    fn add_TOFFOLI_gate(
        slf: PyRefMut<'_, Self>,
        control_index1: usize,
        control_index2: usize,
        target_index: usize,
    ) -> PyResult<()> {
        Self::add_gate(
            slf,
            QuantumGate::TOFFOLI(control_index1, control_index2, target_index),
            None,
        )
    }

    #[allow(non_snake_case)]
    fn add_UnitaryMatrix_gate(
        slf: PyRefMut<'_, Self>,
        target_indices: Vec<usize>,
        unitary_matrix: Vec<Vec<Complex64>>,
    ) -> PyResult<()> {
        Self::add_gate(
            slf,
            crate::circuit::gates::unitary_matrix(target_indices, unitary_matrix)?,
            None,
        )
    }

    #[allow(non_snake_case)]
    fn add_SingleQubitUnitaryMatrix_gate(
        slf: PyRefMut<'_, Self>,
        target_index: usize,
        unitary_matrix: Vec<Vec<Complex64>>,
    ) -> PyResult<()> {
        Self::add_gate(
            slf,
            crate::circuit::gates::single_qubit_unitary_matrix(target_index, unitary_matrix)?,
            None,
        )
    }

    #[allow(non_snake_case)]
    fn add_TwoQubitUnitaryMatrix_gate(
        slf: PyRefMut<'_, Self>,
        target_index1: usize,
        target_index2: usize,
        unitary_matrix: Vec<Vec<Complex64>>,
    ) -> PyResult<()> {
        Self::add_gate(
            slf,
            crate::circuit::gates::two_qubit_unitary_matrix(
                target_index1,
                target_index2,
                unitary_matrix,
            )?,
            None,
        )
    }

    #[allow(non_snake_case)]
    fn add_Pauli_gate(
        slf: PyRefMut<'_, Self>,
        target_indices: Vec<usize>,
        pauli_ids: Vec<u8>,
    ) -> PyResult<()> {
        Self::add_gate(
            slf,
            crate::circuit::gates::pauli(target_indices, pauli_ids),
            None,
        )
    }

    #[allow(non_snake_case)]
    fn add_PauliRotation_gate(
        slf: PyRefMut<'_, Self>,
        target_qubits: Vec<usize>,
        pauli_id_list: Vec<u8>,
        angle: f64,
    ) -> PyResult<()> {
        Self::add_gate(
            slf,
            crate::circuit::gates::pauli_rotation(target_qubits, pauli_id_list, angle),
            None,
        )
    }

    fn measure(
        slf: PyRefMut<'_, Self>,
        qubit_indices: &Bound<'_, PyAny>,
        classical_indices: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let qubit_indices = if let Ok(seq) = qubit_indices.downcast::<PySequence>() {
            (0..(seq.len()?))
                .map(|i| FromPyObject::extract_bound(&seq.get_item(i)?))
                .collect::<PyResult<Vec<_>>>()?
        } else {
            vec![FromPyObject::extract_bound(qubit_indices)?]
        };
        let classical_indices: Vec<usize> =
            if let Ok(seq) = classical_indices.downcast::<PySequence>() {
                (0..(seq.len()?))
                    .map(|i| FromPyObject::extract_bound(&seq.get_item(i)?))
                    .collect::<PyResult<Vec<_>>>()?
            } else {
                vec![FromPyObject::extract_bound(classical_indices)?]
            };

        Self::add_gate(
            slf,
            crate::circuit::gates::measurement(qubit_indices, classical_indices)?,
            None,
        )
    }
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "circuit")?;
    m.add_class::<QuantumCircuit>()?;
    m.add_class::<ImmutableQuantumCircuit>()?;
    Ok(m)
}
