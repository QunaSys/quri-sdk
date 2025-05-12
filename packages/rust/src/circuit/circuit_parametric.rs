use crate::circuit::circuit::ImmutableQuantumCircuit;
use crate::circuit::gate::QuantumGate;
use crate::circuit::parameter::Parameter;
use crate::circuit::MaybeUnbound;
use num_complex::Complex64;
use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;
use pyo3::types::{PyMapping, PySequence};
use quri_parts::BasicBlock;
use std::collections::HashMap;
use std::collections::VecDeque;

#[pyclass(subclass, eq, module = "quri_parts.rust.circuit.circuit_parametric")]
#[derive(Clone)]
#[repr(C)]
pub struct ImmutableParametricQuantumCircuit {
    #[pyo3(get)]
    qubit_count: usize,
    #[pyo3(get)]
    cbit_count: usize,
    gates: BasicBlock<QuantumGate<MaybeUnbound>>,
    depth_cache: Option<usize>,
    is_immutable: bool,
}

impl PartialEq for ImmutableParametricQuantumCircuit {
    fn eq(&self, other: &Self) -> bool {
        &self.qubit_count == &other.qubit_count
            && &self.cbit_count == &other.cbit_count
            && &self.gates.0.len() == &other.gates.0.len()
            && self.gates.0.iter().zip(&other.gates.0).all(|(l, r)| {
                l.clone()
                    .map_param(|p| {
                        if let MaybeUnbound::Bound(p) = p {
                            Some(p)
                        } else {
                            None
                        }
                    })
                    .into_property()
                    == r.clone()
                        .map_param(|p| {
                            if let MaybeUnbound::Bound(p) = p {
                                Some(p)
                            } else {
                                None
                            }
                        })
                        .into_property()
            })
    }
}

impl ImmutableParametricQuantumCircuit {
    fn bind_parameters_internal(
        slf: &Bound<'_, Self>,
        params: Vec<f64>,
    ) -> PyResult<(ImmutableQuantumCircuit, HashMap<Parameter, f64>)> {
        impl MaybeUnbound {
            pub fn unwrap_or_else(&self, f: impl FnOnce(Parameter) -> Option<f64>) -> Option<f64> {
                match self {
                    Self::Bound(p) => Some(*p),
                    Self::Unbound(p) => f(p.clone()),
                }
            }
        }
        let mut map = HashMap::new();
        let mut params: VecDeque<_> = params.into();
        let borrowed = slf.borrow();
        let gates = borrowed
            .gates
            .0
            .iter()
            .map(|gate| {
                Some(match gate {
                    QuantumGate::Identity(q1) => QuantumGate::Identity(*q1),
                    QuantumGate::X(q1) => QuantumGate::X(*q1),
                    QuantumGate::Y(q1) => QuantumGate::Y(*q1),
                    QuantumGate::Z(q1) => QuantumGate::Z(*q1),
                    QuantumGate::H(q1) => QuantumGate::H(*q1),
                    QuantumGate::S(q1) => QuantumGate::S(*q1),
                    QuantumGate::Sdag(q1) => QuantumGate::Sdag(*q1),
                    QuantumGate::SqrtX(q1) => QuantumGate::SqrtX(*q1),
                    QuantumGate::SqrtXdag(q1) => QuantumGate::SqrtXdag(*q1),
                    QuantumGate::SqrtY(q1) => QuantumGate::SqrtY(*q1),
                    QuantumGate::SqrtYdag(q1) => QuantumGate::SqrtYdag(*q1),
                    QuantumGate::T(q1) => QuantumGate::T(*q1),
                    QuantumGate::Tdag(q1) => QuantumGate::Tdag(*q1),
                    QuantumGate::RX(q1, p1) => QuantumGate::RX(
                        *q1,
                        p1.unwrap_or_else(|p| {
                            let f = params.pop_front()?;
                            map.insert(p, f);
                            Some(f)
                        })?,
                    ),
                    QuantumGate::RY(q1, p1) => QuantumGate::RY(
                        *q1,
                        p1.unwrap_or_else(|p| {
                            let f = params.pop_front()?;
                            map.insert(p, f);
                            Some(f)
                        })?,
                    ),
                    QuantumGate::RZ(q1, p1) => QuantumGate::RZ(
                        *q1,
                        p1.unwrap_or_else(|p| {
                            let f = params.pop_front()?;
                            map.insert(p, f);
                            Some(f)
                        })?,
                    ),
                    QuantumGate::U1(q1, p1) => QuantumGate::U1(*q1, *p1),
                    QuantumGate::U2(q1, p1, p2) => QuantumGate::U2(*q1, *p1, *p2),
                    QuantumGate::U3(q1, p1, p2, p3) => QuantumGate::U3(*q1, *p1, *p2, *p3),
                    QuantumGate::CNOT(q1, q2) => QuantumGate::CNOT(*q1, *q2),
                    QuantumGate::CZ(q1, q2) => QuantumGate::CZ(*q1, *q2),
                    QuantumGate::SWAP(q1, q2) => QuantumGate::SWAP(*q1, *q2),
                    QuantumGate::TOFFOLI(q1, q2, q3) => QuantumGate::TOFFOLI(*q1, *q2, *q3),
                    QuantumGate::UnitaryMatrix(q1, mat) => {
                        QuantumGate::UnitaryMatrix(q1.clone(), mat.clone())
                    }
                    QuantumGate::Pauli(q1, pauli_ids) => {
                        QuantumGate::Pauli(q1.clone(), pauli_ids.clone())
                    }
                    QuantumGate::PauliRotation(q1, pauli_ids, p1) => QuantumGate::PauliRotation(
                        q1.clone(),
                        pauli_ids.clone(),
                        p1.unwrap_or_else(|p| {
                            let f = params.pop_front()?;
                            map.insert(p, f);
                            Some(f)
                        })?,
                    ),
                    QuantumGate::Measurement(q1, c1) => {
                        QuantumGate::Measurement(q1.clone(), c1.clone())
                    }
                    QuantumGate::Other(other) => QuantumGate::Other(other.clone()),
                })
            })
            .collect::<Option<_>>();
        if let Some(gates) = gates {
            if params.len() == 0 {
                return Ok((
                    ImmutableQuantumCircuit {
                        qubit_count: borrowed.qubit_count,
                        cbit_count: borrowed.cbit_count,
                        gates: BasicBlock(gates),
                        depth_cache: borrowed.depth_cache.clone(),
                        is_immutable: true,
                    },
                    map,
                ));
            }
        }
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Passed value count ({}) does not match parameter count.",
            params.len(),
        )))
    }
}

#[pymethods]
impl ImmutableParametricQuantumCircuit {
    #[new]
    #[pyo3(signature = (circuit=None))]
    #[pyo3(text_signature = "(circuit: ImmutableParametricQuantumCircuit)")]
    fn py_new(circuit: Option<&Self>) -> Self {
        if let Some(circuit) = circuit {
            ImmutableParametricQuantumCircuit {
                qubit_count: circuit.qubit_count,
                cbit_count: circuit.cbit_count,
                gates: circuit.gates.clone(),
                depth_cache: None,
                is_immutable: true,
            }
        } else {
            // used with __setstate__()
            ImmutableParametricQuantumCircuit {
                qubit_count: 0,
                cbit_count: 0,
                gates: BasicBlock(Vec::new()),
                depth_cache: None,
                is_immutable: true,
            }
        }
    }

    #[pyo3(name = "__getstate__")]
    fn py_getstate(
        slf: Bound<'_, Self>,
    ) -> PyResult<(usize, usize, Vec<(Py<PyAny>, Option<Parameter>)>)> {
        let gates = Self::get_gates_and_params(slf.borrow())?;
        let borrowed = slf.borrow();
        Ok((borrowed.qubit_count, borrowed.cbit_count, gates))
    }

    #[pyo3(name = "__setstate__")]
    fn py_setstate(
        slf: Bound<'_, Self>,
        state: (usize, usize, Vec<(Py<PyAny>, Option<Parameter>)>),
    ) -> PyResult<()> {
        let mut borrowed = slf.borrow_mut();
        let gates = state
            .2
            .into_iter()
            .map(|(gate, param)| {
                if let Ok(gate) = gate.extract::<'_, '_, QuantumGate<f64>>(slf.py()) {
                    Ok(gate.map_param(MaybeUnbound::Bound))
                } else {
                    let gate = gate.extract::<'_, '_, QuantumGate<()>>(slf.py())?;
                    let param = param.ok_or(pyo3::exceptions::PyValueError::new_err(format!(
                        "Requires parameter",
                    )))?;
                    Ok(gate.map_param(|()| MaybeUnbound::Unbound(param.clone())))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        borrowed.gates = quri_parts::BasicBlock(gates);
        borrowed.qubit_count = state.0;
        borrowed.cbit_count = state.1;
        borrowed.depth_cache = None;
        Ok(())
    }

    #[getter]
    fn get_gates<'py>(slf: PyRef<'py, Self>) -> PyResult<Vec<PyObject>> {
        slf.gates
            .0
            .iter()
            .map(|g| match g.clone().instantiate()? {
                Ok(g) => Ok(QuantumGate::into_py(g, slf.py())),
                Err(g) => Ok(g.0.into_py(slf.py()).into_any()),
            })
            .collect()
    }

    #[getter]
    fn get_gates_and_params(slf: PyRef<'_, Self>) -> PyResult<Vec<(Py<PyAny>, Option<Parameter>)>> {
        slf.gates
            .0
            .iter()
            .map(|g| match g.clone().instantiate()? {
                Ok(g) => Ok((QuantumGate::into_py(g, slf.py()), None)),
                Err(g) => Ok((g.0.into_py(slf.py()).into_any(), Some(g.1))),
            })
            .collect()
    }

    /// For compatibility with
    /// quri_parts.circuit.circuit_parametric.ImmutableBoundParametricQuantumCircuit
    #[allow(non_snake_case)]
    #[getter]
    fn get__gates(slf: PyRef<'_, Self>) -> PyResult<Vec<(Py<PyAny>, Option<Parameter>)>> {
        Self::get_gates_and_params(slf)
    }

    #[getter]
    fn get_has_trivial_parameter_mapping(&self) -> bool {
        true
    }

    #[getter]
    fn get_depth(&mut self) -> usize {
        let depth = &mut self.depth_cache;
        if let Some(depth) = *depth {
            depth
        } else {
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
            let d = ds.into_values().max().unwrap_or(0);
            *depth = Some(d);
            d
        }
    }

    #[getter]
    fn get_param_mapping<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let params = Self::get__params(slf.borrow());
        slf.py()
            .import_bound("quri_parts.circuit.parameter_mapping")?
            .getattr("LinearParameterMapping")?
            .call1((
                params.clone(),
                params.clone(),
                params
                    .iter()
                    .map(|p| (p.clone(), p.clone()))
                    .collect::<HashMap<_, _>>(),
            ))
    }

    fn primitive_circuit(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        Self::freeze(slf)
    }

    fn get_mutable_copy(slf: &Bound<'_, Self>) -> PyResult<Py<ParametricQuantumCircuit>> {
        let mut cloned = slf.borrow().clone();
        cloned.is_immutable = false;
        Py::new(slf.py(), (ParametricQuantumCircuit(), cloned))
    }

    #[pyo3(
        text_signature = "(gates: ImmutableQuantumCircuit | ImmutableParametricQuantumCircuit | Sequence[QuantumGate])"
    )]
    fn combine(
        slf: &Bound<'_, Self>,
        gates: &Bound<'_, PyAny>,
    ) -> PyResult<Py<ParametricQuantumCircuit>> {
        let ret = Self::get_mutable_copy(slf)?;
        ParametricQuantumCircuit::extend(ret.bind(slf.py()).clone(), gates)?;
        Ok(ret)
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

    fn bind_parameters(
        slf: &Bound<'_, Self>,
        params: Vec<f64>,
    ) -> PyResult<Py<ImmutableBoundParametricQuantumCircuit>> {
        let (ins, map) = Self::bind_parameters_internal(slf, params)?;
        Py::new(
            slf.py(),
            (
                ImmutableBoundParametricQuantumCircuit(map, slf.clone().unbind()),
                ins,
            ),
        )
    }

    fn bind_parameters_by_dict(
        slf: &Bound<'_, Self>,
        params_dict: HashMap<Parameter, f64>,
    ) -> PyResult<Py<ImmutableBoundParametricQuantumCircuit>> {
        let params = Self::get__params(slf.borrow());
        Self::bind_parameters(
            slf,
            params
                .iter()
                .map(|p| {
                    params_dict.get(p).cloned().ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Cannot find parameter {} in params_dict",
                            p
                        ))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?,
        )
    }

    #[allow(non_snake_case)]
    #[getter]
    fn get__params(slf: PyRef<'_, Self>) -> Vec<Parameter> {
        slf.gates
            .0
            .iter()
            .map(|g| g.get_params())
            .flatten()
            .filter_map(|g| {
                if let MaybeUnbound::Unbound(p) = g {
                    Some(p)
                } else {
                    None
                }
            })
            .collect()
    }

    #[getter]
    fn parameter_count(slf: PyRef<'_, Self>) -> usize {
        Self::get__params(slf).len()
    }

    #[pyo3(name = "__add__")]
    fn py_add(slf: &Bound<'_, Self>, gates: &Bound<'_, PyAny>) -> PyObject {
        Self::combine(slf, gates)
            .map(|c| c.into_py(slf.py()))
            .unwrap_or(slf.py().NotImplemented())
    }

    #[pyo3(name = "__radd__")]
    fn py_radd(
        slf: &Bound<'_, Self>,
        gates: &Bound<'_, PyAny>,
    ) -> PyResult<Py<ParametricQuantumCircuit>> {
        let ret = Py::new(
            slf.py(),
            (
                ParametricQuantumCircuit(),
                Self {
                    qubit_count: slf.borrow().qubit_count,
                    cbit_count: 0,
                    gates: BasicBlock(Vec::new()),
                    depth_cache: None,
                    is_immutable: false,
                },
            ),
        )?;
        ParametricQuantumCircuit::extend(ret.bind(slf.py()).clone(), gates)?;
        ParametricQuantumCircuit::extend(ret.bind(slf.py()).clone(), slf.as_any())?;
        Ok(ret)
    }

    fn sample<'py>(
        slf: &Bound<'py, Self>,
        shot_count: i32,
        params: Vec<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sampling =
            PyModule::import_bound(slf.py(), "quri_parts.core.sampling.default_sampler")?;
        let sampling_counts = sampling
            .getattr("DEFAULT_SAMPLER")?
            .call1((slf, shot_count, params));
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
    extends=ImmutableParametricQuantumCircuit,
    subclass,
    module = "quri_parts.rust.circuit.circuit_parametric"
)]
#[derive(Clone, Debug)]
pub struct ParametricQuantumCircuit();

impl ParametricQuantumCircuit {
    fn add_gate_inner(
        slf: &mut PyRefMut<'_, Self>,
        gate: QuantumGate<MaybeUnbound>,
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
                gates.insert(gate_index, gate);
                Ok(())
            } else {
                Err(pyo3::exceptions::PyIndexError::new_err(""))
            }
        } else {
            gates.push(gate);
            Ok(())
        }
    }
}

#[pymethods]
impl ParametricQuantumCircuit {
    #[new]
    #[pyo3(signature = (qubit_count=0, cbit_count=0))]
    #[pyo3(text_signature = "(qubit_count: int, cbit_count: int = 0)")]
    fn py_new(qubit_count: usize, cbit_count: usize) -> (Self, ImmutableParametricQuantumCircuit) {
        let base = ImmutableParametricQuantumCircuit {
            qubit_count,
            cbit_count,
            gates: BasicBlock(vec![].into()),
            depth_cache: None,
            is_immutable: false,
        };
        (ParametricQuantumCircuit(), base)
    }

    #[pyo3(signature = (gate, gate_index = None))]
    #[pyo3(text_signature = "(gate: QuantumGate, gate_index: Optional[int])")]
    fn add_gate(
        mut slf: PyRefMut<'_, Self>,
        gate: QuantumGate,
        gate_index: Option<usize>,
    ) -> PyResult<()> {
        let gate = gate.map_param(|p| MaybeUnbound::Bound(p));
        Self::add_gate_inner(&mut slf, gate, gate_index)
    }

    #[pyo3(
        text_signature = "(gates: ImmutableParametricQuantumCircuit | ImmutableQuantumCircuit | Sequence[QuantumGate])"
    )]
    fn extend(slf: Bound<'_, Self>, gates: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(other) = gates.downcast::<ImmutableQuantumCircuit>() {
            for gate in other.borrow().gates.0.iter() {
                Self::add_gate(slf.borrow_mut(), gate.clone(), None)?;
            }
            Ok(())
        } else if let Ok(other) = gates.downcast::<ImmutableParametricQuantumCircuit>() {
            for gate in other.borrow().gates.0.iter() {
                Self::add_gate_inner(&mut slf.borrow_mut(), gate.clone(), None)?;
            }
            Ok(())
        } else if let Ok(other) = gates.downcast::<PySequence>() {
            for i in 0..other.len()? {
                let item = other.get_item(i)?;
                let gate = QuantumGate::extract_bound(&item)?;
                Self::add_gate(slf.borrow_mut(), gate, None)?;
            }
            Ok(())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Bad argument `gates`",
            ))
        }
    }

    #[pyo3(name = "__iadd__")]
    fn py_iadd(slf: Bound<'_, Self>, gates: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::extend(slf, gates).map_err(|_| pyo3::exceptions::PyNotImplementedError::new_err(""))
    }

    #[allow(non_snake_case)]
    fn add_ParametricRX_gate(
        mut slf: PyRefMut<'_, Self>,
        qubit_index: usize,
    ) -> PyResult<Parameter> {
        let param = Parameter::new(String::new());
        Self::add_gate_inner(
            &mut slf,
            QuantumGate::RX(qubit_index, MaybeUnbound::Unbound(param.clone())),
            None,
        )?;
        Ok(param)
    }

    #[allow(non_snake_case)]
    fn add_ParametricRY_gate(
        mut slf: PyRefMut<'_, Self>,
        qubit_index: usize,
    ) -> PyResult<Parameter> {
        let param = Parameter::new(String::new());
        Self::add_gate_inner(
            &mut slf,
            QuantumGate::RY(qubit_index, MaybeUnbound::Unbound(param.clone())),
            None,
        )?;
        Ok(param)
    }

    #[allow(non_snake_case)]
    fn add_ParametricRZ_gate(
        mut slf: PyRefMut<'_, Self>,
        qubit_index: usize,
    ) -> PyResult<Parameter> {
        let param = Parameter::new(String::new());
        Self::add_gate_inner(
            &mut slf,
            QuantumGate::RZ(qubit_index, MaybeUnbound::Unbound(param.clone())),
            None,
        )?;
        Ok(param)
    }

    #[allow(non_snake_case)]
    fn add_ParametricPauliRotation_gate(
        mut slf: PyRefMut<'_, Self>,
        target_indices: Vec<usize>,
        pauli_ids: Vec<u8>,
    ) -> PyResult<Parameter> {
        let param = Parameter::new(String::new());
        Self::add_gate_inner(
            &mut slf,
            QuantumGate::PauliRotation(
                target_indices.into(),
                pauli_ids.into(),
                MaybeUnbound::Unbound(param.clone()),
            ),
            None,
        )?;
        Ok(param)
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

#[pyclass(
    extends=ImmutableQuantumCircuit,
    subclass,
    module = "quri_parts.rust.circuit.circuit_parametric"
)]
#[derive(Debug)]
pub struct ImmutableBoundParametricQuantumCircuit(
    HashMap<Parameter, f64>,
    Py<ImmutableParametricQuantumCircuit>,
);

#[pymethods]
impl ImmutableBoundParametricQuantumCircuit {
    #[new]
    #[pyo3(signature = (circuit, parameter_map))]
    fn py_new<'py>(
        circuit: Bound<'py, ImmutableParametricQuantumCircuit>,
        parameter_map: Bound<'py, PyMapping>,
    ) -> PyResult<Py<Self>> {
        let map = parameter_map
            .items()?
            .unbind()
            .extract::<'_, '_, Vec<(Parameter, f64)>>(circuit.py())?
            .into_iter()
            .collect();
        ImmutableParametricQuantumCircuit::bind_parameters_by_dict(&circuit, map)
    }

    #[pyo3(name = "__eq__")]
    fn py_eq<'py>(slf: PyRef<'_, Self>, other: PyRef<'_, Self>) -> bool {
        slf.as_super().eq(other.as_super())
    }

    #[getter]
    fn get_unbound_param_circuit(&self) -> Py<ImmutableParametricQuantumCircuit> {
        self.1.clone()
    }

    #[getter]
    fn get_parameter_map(&self) -> HashMap<Parameter, f64> {
        self.0.clone()
    }

    fn freeze(slf: Py<Self>) -> Py<Self> {
        slf
    }
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "circuit_parametric")?;
    m.add_class::<ImmutableParametricQuantumCircuit>()?;
    m.add_class::<ImmutableBoundParametricQuantumCircuit>()?;
    m.add_class::<ParametricQuantumCircuit>()?;
    Ok(m)
}
