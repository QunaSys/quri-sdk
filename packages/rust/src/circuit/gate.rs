use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
pub enum QuantumGate<P = f64> {
    Identity(usize),
    X(usize),
    Y(usize),
    Z(usize),
    H(usize),
    S(usize),
    Sdag(usize),
    SqrtX(usize),
    SqrtXdag(usize),
    SqrtY(usize),
    SqrtYdag(usize),
    T(usize),
    Tdag(usize),
    RX(usize, P),
    RY(usize, P),
    RZ(usize, P),
    U1(usize, f64),
    U2(usize, f64, f64),
    U3(usize, f64, f64, f64),
    CNOT(usize, usize),
    CZ(usize, usize),
    SWAP(usize, usize),
    TOFFOLI(usize, usize, usize),
    UnitaryMatrix(Vec<usize>, Vec<Vec<Complex64>>),
    Pauli(Vec<usize>, Vec<u8>),
    PauliRotation(Vec<usize>, Vec<u8>, P),
    Measurement(Vec<usize>, Vec<usize>),
    Other(Box<GenericGateProperty>),
}

impl<P: Clone> QuantumGate<P> {
    pub fn get_qubits(&self) -> Vec<usize> {
        match self {
            Self::Identity(q)
            | Self::X(q)
            | Self::Y(q)
            | Self::Z(q)
            | Self::H(q)
            | Self::S(q)
            | Self::Sdag(q)
            | Self::SqrtX(q)
            | Self::SqrtXdag(q)
            | Self::SqrtY(q)
            | Self::SqrtYdag(q)
            | Self::T(q)
            | Self::Tdag(q)
            | Self::RX(q, _)
            | Self::RY(q, _)
            | Self::RZ(q, _)
            | Self::U1(q, _)
            | Self::U2(q, _, _)
            | Self::U3(q, _, _, _) => vec![*q],
            Self::Measurement(qs, _) => qs.clone().into(),
            Self::CNOT(q0, q1) | Self::CZ(q0, q1) | Self::SWAP(q0, q1) => vec![*q0, *q1],
            Self::TOFFOLI(q0, q1, q2) => vec![*q0, *q1, *q2],
            Self::UnitaryMatrix(qs, _) | Self::Pauli(qs, _) | Self::PauliRotation(qs, _, _) => {
                qs.clone().into()
            }
            Self::Other(o) => {
                let mut ret = o.control_indices.clone();
                ret.extend(o.target_indices.clone());
                ret.into()
            }
        }
    }

    pub fn get_cbits(&self) -> Vec<usize> {
        match self {
            Self::Measurement(_, cs) => cs.clone().into(),
            Self::Other(o) => o.classical_indices.clone().into(),
            _ => vec![],
        }
    }

    pub fn get_params(&self) -> Vec<P> {
        match self {
            Self::RX(_, p) | Self::RY(_, p) | Self::RZ(_, p) => vec![p.clone()],
            Self::PauliRotation(_, _, p) => vec![p.clone()],
            _ => vec![],
        }
    }

    pub fn map_param<Q>(self, mut f: impl FnMut(P) -> Q) -> QuantumGate<Q> {
        match self {
            QuantumGate::Identity(q) => QuantumGate::Identity(q),
            QuantumGate::X(q) => QuantumGate::X(q),
            QuantumGate::Y(q) => QuantumGate::Y(q),
            QuantumGate::Z(q) => QuantumGate::Z(q),
            QuantumGate::H(q) => QuantumGate::H(q),
            QuantumGate::S(q) => QuantumGate::S(q),
            QuantumGate::Sdag(q) => QuantumGate::Sdag(q),
            QuantumGate::SqrtX(q) => QuantumGate::SqrtX(q),
            QuantumGate::SqrtXdag(q) => QuantumGate::SqrtXdag(q),
            QuantumGate::SqrtY(q) => QuantumGate::SqrtY(q),
            QuantumGate::SqrtYdag(q) => QuantumGate::SqrtYdag(q),
            QuantumGate::T(q) => QuantumGate::T(q),
            QuantumGate::Tdag(q) => QuantumGate::Tdag(q),
            QuantumGate::RX(q, p) => QuantumGate::RX(q, f(p)),
            QuantumGate::RY(q, p) => QuantumGate::RY(q, f(p)),
            QuantumGate::RZ(q, p) => QuantumGate::RZ(q, f(p)),
            QuantumGate::U1(q, p) => QuantumGate::U1(q, p),
            QuantumGate::U2(q, p0, p1) => QuantumGate::U2(q, p0, p1),
            QuantumGate::U3(q, p0, p1, p2) => QuantumGate::U3(q, p0, p1, p2),
            QuantumGate::CNOT(q0, q1) => QuantumGate::CNOT(q0, q1),
            QuantumGate::CZ(q0, q1) => QuantumGate::CZ(q0, q1),
            QuantumGate::SWAP(q0, q1) => QuantumGate::SWAP(q0, q1),
            QuantumGate::TOFFOLI(q0, q1, q2) => QuantumGate::TOFFOLI(q0, q1, q2),
            QuantumGate::UnitaryMatrix(qs, mat) => QuantumGate::UnitaryMatrix(qs, mat),
            QuantumGate::Pauli(qs, ps) => QuantumGate::Pauli(qs, ps),
            QuantumGate::PauliRotation(qs, ps, a) => QuantumGate::PauliRotation(qs, ps, f(a)),
            QuantumGate::Measurement(qs, cs) => QuantumGate::Measurement(qs, cs),
            QuantumGate::Other(o) => QuantumGate::Other(o),
        }
    }
}

impl QuantumGate<Option<f64>> {
    pub fn into_property(self) -> GenericGateProperty {
        fn nonpara_named_1q_gate(name: &str, q: usize) -> GenericGateProperty {
            GenericGateProperty {
                name: name.to_owned().into(),
                target_indices: vec![q].into(),
                control_indices: vec![].into(),
                classical_indices: vec![].into(),
                params: vec![].into(),
                pauli_ids: vec![].into(),
                unitary_matrix: None,
            }
        }

        fn para_1q_gate(name: &str, q: usize, p: f64) -> GenericGateProperty {
            GenericGateProperty {
                name: name.to_owned().into(),
                target_indices: vec![q].into(),
                control_indices: vec![].into(),
                classical_indices: vec![].into(),
                params: vec![p].into(),
                pauli_ids: vec![].into(),
                unitary_matrix: None,
            }
        }

        match self {
            Self::Identity(q) => nonpara_named_1q_gate("Identity", q),
            Self::X(q) => nonpara_named_1q_gate("X", q),
            Self::Y(q) => nonpara_named_1q_gate("Y", q),
            Self::Z(q) => nonpara_named_1q_gate("Z", q),
            Self::H(q) => nonpara_named_1q_gate("H", q),
            Self::S(q) => nonpara_named_1q_gate("S", q),
            Self::Sdag(q) => nonpara_named_1q_gate("Sdag", q),
            Self::SqrtX(q) => nonpara_named_1q_gate("SqrtX", q),
            Self::SqrtXdag(q) => nonpara_named_1q_gate("SqrtXdag", q),
            Self::SqrtY(q) => nonpara_named_1q_gate("SqrtY", q),
            Self::SqrtYdag(q) => nonpara_named_1q_gate("SqrtYdag", q),
            Self::T(q) => nonpara_named_1q_gate("T", q),
            Self::Tdag(q) => nonpara_named_1q_gate("Tdag", q),
            Self::RX(q, Some(p)) => para_1q_gate("RX", q, p),
            Self::RY(q, Some(p)) => para_1q_gate("RY", q, p),
            Self::RZ(q, Some(p)) => para_1q_gate("RZ", q, p),
            Self::RX(q, None) => nonpara_named_1q_gate("ParametricRX", q),
            Self::RY(q, None) => nonpara_named_1q_gate("ParametricRY", q),
            Self::RZ(q, None) => nonpara_named_1q_gate("ParametricRZ", q),
            Self::U1(q, p) => para_1q_gate("U1", q, p),
            Self::U2(q, p0, p1) => GenericGateProperty {
                name: "U2".to_owned().into(),
                target_indices: vec![q].into(),
                params: vec![p0, p1].into(),
                ..Default::default()
            },
            Self::U3(q, p0, p1, p2) => GenericGateProperty {
                name: "U3".to_owned().into(),
                target_indices: vec![q].into(),
                params: vec![p0, p1, p2].into(),
                ..Default::default()
            },
            Self::CNOT(q0, q1) => GenericGateProperty {
                name: "CNOT".to_owned().into(),
                control_indices: vec![q0].into(),
                target_indices: vec![q1].into(),
                ..Default::default()
            },
            Self::CZ(q0, q1) => GenericGateProperty {
                name: "CZ".to_owned().into(),
                control_indices: vec![q0].into(),
                target_indices: vec![q1].into(),
                ..Default::default()
            },
            Self::SWAP(q0, q1) => GenericGateProperty {
                name: "SWAP".to_owned().into(),
                target_indices: vec![q0, q1].into(),
                ..Default::default()
            },
            Self::TOFFOLI(q0, q1, q2) => GenericGateProperty {
                name: "TOFFOLI".to_owned().into(),
                control_indices: vec![q0, q1].into(),
                target_indices: vec![q2].into(),
                ..Default::default()
            },
            Self::UnitaryMatrix(target_indices, mat) => GenericGateProperty {
                name: "UnitaryMatrix".to_owned().into(),
                target_indices,
                unitary_matrix: Some(mat).into(),
                ..Default::default()
            },
            Self::Pauli(target_indices, pauli_ids) => GenericGateProperty {
                name: "Pauli".to_owned().into(),
                target_indices,
                pauli_ids,
                ..Default::default()
            },
            Self::PauliRotation(target_indices, pauli_ids, Some(angle)) => GenericGateProperty {
                name: "PauliRotation".to_owned().into(),
                target_indices,
                params: vec![angle].into(),
                pauli_ids,
                ..Default::default()
            },
            Self::PauliRotation(target_indices, pauli_ids, None) => GenericGateProperty {
                name: "ParametricPauliRotation".to_owned().into(),
                target_indices,
                pauli_ids,
                ..Default::default()
            },
            Self::Measurement(target_indices, classical_indices) => GenericGateProperty {
                name: "Measurement".to_owned().into(),
                target_indices,
                classical_indices,
                ..Default::default()
            },
            Self::Other(o) => *o,
        }
    }
}

#[derive(Clone, Debug, Default)]
#[repr(C)]
pub struct GenericGateProperty {
    pub name: String,
    pub target_indices: Vec<usize>,
    pub control_indices: Vec<usize>,
    pub classical_indices: Vec<usize>,
    pub params: Vec<f64>,
    pub pauli_ids: Vec<u8>,
    pub unitary_matrix: Option<Vec<Vec<Complex64>>>,
}

impl GenericGateProperty {
    pub fn into_gate(self) -> PyResult<QuantumGate<Option<f64>>> {
        let is_nonpara_named_1q_gate = || {
            self.target_indices.len() == 1
                && self.control_indices.is_empty()
                && self.classical_indices.is_empty()
                && self.params.is_empty()
                && self.pauli_ids.is_empty()
                && self.unitary_matrix.is_none()
        };

        let is_para_1q_gate = || {
            self.target_indices.len() == 1
                && self.control_indices.is_empty()
                && self.classical_indices.is_empty()
                && self.params.len() == 1
                && self.pauli_ids.is_empty()
                && self.unitary_matrix.is_none()
        };

        match self.name.as_str() {
            "Identity" if is_nonpara_named_1q_gate() => {
                Ok(QuantumGate::Identity(self.target_indices[0]))
            }
            "X" if is_nonpara_named_1q_gate() => Ok(QuantumGate::X(self.target_indices[0])),
            "Y" if is_nonpara_named_1q_gate() => Ok(QuantumGate::Y(self.target_indices[0])),
            "Z" if is_nonpara_named_1q_gate() => Ok(QuantumGate::Z(self.target_indices[0])),
            "H" if is_nonpara_named_1q_gate() => Ok(QuantumGate::H(self.target_indices[0])),
            "S" if is_nonpara_named_1q_gate() => Ok(QuantumGate::S(self.target_indices[0])),
            "Sdag" if is_nonpara_named_1q_gate() => Ok(QuantumGate::Sdag(self.target_indices[0])),
            "SqrtX" if is_nonpara_named_1q_gate() => Ok(QuantumGate::SqrtX(self.target_indices[0])),
            "SqrtXdag" if is_nonpara_named_1q_gate() => {
                Ok(QuantumGate::SqrtXdag(self.target_indices[0]))
            }
            "SqrtY" if is_nonpara_named_1q_gate() => Ok(QuantumGate::SqrtY(self.target_indices[0])),
            "SqrtYdag" if is_nonpara_named_1q_gate() => {
                Ok(QuantumGate::SqrtYdag(self.target_indices[0]))
            }
            "T" if is_nonpara_named_1q_gate() => Ok(QuantumGate::T(self.target_indices[0])),
            "Tdag" if is_nonpara_named_1q_gate() => Ok(QuantumGate::Tdag(self.target_indices[0])),
            "RX" if is_para_1q_gate() => Ok(QuantumGate::RX(
                self.target_indices[0],
                Some(self.params[0]),
            )),
            "RY" if is_para_1q_gate() => Ok(QuantumGate::RY(
                self.target_indices[0],
                Some(self.params[0]),
            )),
            "RZ" if is_para_1q_gate() => Ok(QuantumGate::RZ(
                self.target_indices[0],
                Some(self.params[0]),
            )),
            "U1" if is_para_1q_gate() => {
                Ok(QuantumGate::U1(self.target_indices[0], self.params[0]))
            }
            "U2" if self.control_indices.is_empty()
                && self.target_indices.len() == 1
                && self.classical_indices.is_empty()
                && self.params.len() == 2
                && self.pauli_ids.is_empty()
                && self.unitary_matrix.is_none() =>
            {
                Ok(QuantumGate::U2(
                    self.target_indices[0],
                    self.params[0],
                    self.params[1],
                ))
            }
            "U3" if self.control_indices.is_empty()
                && self.target_indices.len() == 1
                && self.classical_indices.is_empty()
                && self.params.len() == 3
                && self.pauli_ids.is_empty()
                && self.unitary_matrix.is_none() =>
            {
                Ok(QuantumGate::U3(
                    self.target_indices[0],
                    self.params[0],
                    self.params[1],
                    self.params[2],
                ))
            }
            "CNOT"
                if self.control_indices.len() == 1
                    && self.target_indices.len() == 1
                    && self.classical_indices.is_empty()
                    && self.params.is_empty()
                    && self.pauli_ids.is_empty()
                    && self.unitary_matrix.is_none() =>
            {
                Ok(QuantumGate::CNOT(
                    self.control_indices[0],
                    self.target_indices[0],
                ))
            }
            "CZ" if self.control_indices.len() == 1
                && self.target_indices.len() == 1
                && self.classical_indices.is_empty()
                && self.params.is_empty()
                && self.pauli_ids.is_empty()
                && self.unitary_matrix.is_none() =>
            {
                Ok(QuantumGate::CZ(
                    self.control_indices[0],
                    self.target_indices[0],
                ))
            }
            "SWAP"
                if self.control_indices.is_empty()
                    && self.target_indices.len() == 2
                    && self.classical_indices.is_empty()
                    && self.params.is_empty()
                    && self.pauli_ids.is_empty()
                    && self.unitary_matrix.is_none() =>
            {
                Ok(QuantumGate::SWAP(
                    self.target_indices[0],
                    self.target_indices[1],
                ))
            }
            "TOFFOLI"
                if self.control_indices.len() == 2
                    && self.target_indices.len() == 1
                    && self.classical_indices.is_empty()
                    && self.params.is_empty()
                    && self.pauli_ids.is_empty()
                    && self.unitary_matrix.is_none() =>
            {
                Ok(QuantumGate::TOFFOLI(
                    self.control_indices[0],
                    self.control_indices[1],
                    self.target_indices[0],
                ))
            }
            "UnitaryMatrix"
                if self.control_indices.is_empty()
                    && self.classical_indices.is_empty()
                    && self.params.is_empty()
                    && self.pauli_ids.is_empty()
                    && self.unitary_matrix.is_some() =>
            {
                Ok(crate::circuit::gates::unitary_matrix(
                    self.target_indices.into(),
                    self.unitary_matrix
                        .unwrap()
                        .into_iter()
                        .map(|v| v.into())
                        .collect(),
                )?
                .map_param(|_| unreachable!()))
            }
            "Pauli"
                if self.control_indices.is_empty()
                    && self.classical_indices.is_empty()
                    && self.params.is_empty()
                    && self.pauli_ids.len() == self.target_indices.len()
                    && self.unitary_matrix.is_none() =>
            {
                Ok(QuantumGate::Pauli(self.target_indices, self.pauli_ids))
            }
            "PauliRotation"
                if self.control_indices.is_empty()
                    && self.classical_indices.is_empty()
                    && self.params.len() == 1
                    && self.pauli_ids.len() == self.target_indices.len()
                    && self.unitary_matrix.is_none() =>
            {
                Ok(QuantumGate::PauliRotation(
                    self.target_indices,
                    self.pauli_ids,
                    Some(self.params[0]),
                ))
            }
            "ParametricRX" if is_nonpara_named_1q_gate() => {
                Ok(QuantumGate::RX(self.target_indices[0], None))
            }
            "ParametricRY" if is_nonpara_named_1q_gate() => {
                Ok(QuantumGate::RY(self.target_indices[0], None))
            }
            "ParametricRZ" if is_nonpara_named_1q_gate() => {
                Ok(QuantumGate::RZ(self.target_indices[0], None))
            }
            "Measurement"
                if self.control_indices.is_empty()
                    && self.target_indices.len() == self.classical_indices.len()
                    && self.params.is_empty()
                    && self.pauli_ids.is_empty()
                    && self.unitary_matrix.is_none() =>
            {
                Ok(QuantumGate::Measurement(
                    self.target_indices,
                    self.classical_indices,
                ))
            }

            _ => Ok(QuantumGate::Other(Box::new(self))),
        }
    }
}

fn unordered_eq<I0, I1>(lhs: I0, rhs: I1) -> bool
where
    I0: IntoIterator,
    I1: IntoIterator,
    I0::IntoIter: ExactSizeIterator + Clone,
    I1::IntoIter: ExactSizeIterator<Item = I0::Item> + Clone,
    I0::Item: PartialEq,
    I1::Item: Clone,
{
    let lhs = lhs.into_iter();
    let mut rhs = rhs.into_iter();
    if lhs.len() != rhs.len() {
        return false;
    }
    lhs.clone().all(|q| rhs.clone().find(|r| *r == q).is_some())
        && rhs.all(|q| lhs.clone().find(|r| *r == q).is_some())
}

#[test]
fn test_unordered_eq() {
    assert!(unordered_eq(&[0, 1], &[1, 0]));
}

impl PartialEq for GenericGateProperty {
    fn eq(&self, other: &Self) -> bool {
        if &self.name != &other.name
            || !unordered_eq(&self.control_indices, &other.control_indices)
            || &self.params != &other.params
        {
            return false;
        }
        if &self.name == "UnitaryMatrix" {
            if &self.target_indices != &other.target_indices {
                return false;
            }
        } else {
            if !unordered_eq(&self.target_indices, &other.target_indices) {
                return false;
            }
        }
        if !unordered_eq(
            self.target_indices.iter().zip(&self.pauli_ids),
            other.target_indices.iter().zip(&other.pauli_ids),
        ) {
            return false;
        }
        if &self.name == "Measurement" {
            if !unordered_eq(
                self.target_indices.iter().zip(&self.classical_indices),
                other.target_indices.iter().zip(&other.classical_indices),
            ) {
                return false;
            }
        }
        &self.unitary_matrix == &other.unitary_matrix
    }
}

pub type ParametricQuantumGate = QuantumGate<()>;

fn format_tuple<T: core::fmt::Display>(input: &[T]) -> String {
    let out = input
        .iter()
        .map(|i| format!("{}", i))
        .collect::<Vec<_>>()
        .join(", ");
    if input.len() == 1 {
        out + ","
    } else {
        out
    }
}

impl GenericGateProperty {
    pub fn get_compat_string(&self) -> String {
        format!("QuantumGate(name='{}', target_indices=({}), control_indices=({}), classical_indices=({}), params=({}), pauli_ids=({}), unitary_matrix={})",
            &self.name,
            format_tuple(self.target_indices.as_slice()),
            format_tuple(self.control_indices.as_slice()),
            format_tuple(self.classical_indices.as_slice()),
            format_tuple(self.params.as_slice()),
            format_tuple(self.pauli_ids.as_slice()),
            {
                if let Some(matrix) = &self.unitary_matrix {
                    format_tuple(matrix.iter().map(|row| format!("({})", format_tuple(row))).collect::<Vec<_>>().as_slice())
                }else {"()".to_owned()}
            }
        )
    }
    pub fn get_compat_string_parametric(&self) -> String {
        format!("ParametricQuantumGate(name='{}', target_indices=({}), control_indices=({}), pauli_ids=({}))",
            &self.name,
            format_tuple(self.target_indices.as_slice()),
            format_tuple(self.control_indices.as_slice()),
            format_tuple(self.pauli_ids.as_slice()),
        )
    }
}

mod wrapper {
    use super::*;
    #[pyclass(frozen, module = "quri_parts.rust.circuit.gate", name = "QuantumGate")]
    #[derive(Clone, Debug, PartialEq)]
    struct QuantumGateWrapper(QuantumGate<f64>);

    impl<T> IntoPy<T> for QuantumGate<f64>
    where
        QuantumGateWrapper: IntoPy<T>,
    {
        fn into_py(self, py: Python<'_>) -> T {
            QuantumGateWrapper(self).into_py(py)
        }
    }

    impl<'py> pyo3::conversion::FromPyObject<'py> for QuantumGate<f64> {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            Ok(<QuantumGateWrapper as pyo3::conversion::FromPyObject<'py>>::extract_bound(ob)?.0)
        }
    }

    #[pymethods]
    impl QuantumGateWrapper {
        #[new]
        #[pyo3(
            signature = (name, target_indices, control_indices=Vec::new(), classical_indices=Vec::new(), params=Vec::new(), pauli_ids=Vec::new(), unitary_matrix=None),
            text_signature = "(
                name: str,
                target_indices: Sequence[int],
                control_indices: Sequence[int] = [],
                classical_indices: Sequence[int] = [],
                params: Sequence[float] = [],
                pauli_ids: Sequence[int] = [],
                unitary_matrix: Optional[Sequence[Sequence[complex]]] = None
            )"
        )]
        fn py_new(
            name: String,
            target_indices: Vec<usize>,
            control_indices: Vec<usize>,
            classical_indices: Vec<usize>,
            params: Vec<f64>,
            pauli_ids: Vec<u8>,
            mut unitary_matrix: Option<Vec<Vec<Complex64>>>,
        ) -> PyResult<Self> {
            if let Some(matrix) = &unitary_matrix {
                if matrix.len() == 0 {
                    unitary_matrix = None;
                }
            }
            let prop = GenericGateProperty {
                name: name.clone().into(),
                target_indices: target_indices.into(),
                control_indices: control_indices.into(),
                classical_indices: classical_indices.into(),
                params: params.into(),
                pauli_ids: pauli_ids.into(),
                unitary_matrix: unitary_matrix
                    .map(|v| v.into_iter().map(Into::into).collect())
                    .into(),
            };
            let mut bad_flag = false;
            let ret = prop.into_gate()?.map_param(|p| match p {
                Some(p) => p,
                None => {
                    bad_flag = true;
                    0.0
                }
            });
            if bad_flag {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Cannot initialize QuantumGate with {}",
                    &name
                )))
            } else {
                Ok(Self(ret))
            }
        }

        #[pyo3(name = "__repr__")]
        fn py_repr(&self) -> String {
            self.0
                .clone()
                .map_param(Some)
                .into_property()
                .get_compat_string()
        }

        #[pyo3(name = "__reduce__")]
        fn py_reduce(
            slf: &Bound<'_, Self>,
        ) -> PyResult<(
            PyObject,
            (
                String,
                Vec<usize>,
                Vec<usize>,
                Vec<usize>,
                Vec<f64>,
                Vec<u8>,
                Option<Vec<Vec<Complex64>>>,
            ),
        )> {
            let data = &slf.get().0.clone().map_param(Some).into_property();
            Ok((
                slf.getattr("__class__").unwrap().unbind(),
                (
                    data.name.clone().into(),
                    data.target_indices.clone().into(),
                    data.control_indices.clone().into(),
                    data.classical_indices.clone().into(),
                    data.params.clone().into(),
                    data.pauli_ids.clone().into(),
                    data.unitary_matrix
                        .clone()
                        .map(|v| v.into_iter().map(Into::into).collect())
                        .into(),
                ),
            ))
        }

        #[pyo3(name = "__hash__")]
        pub(crate) fn py_hash(&self) -> u64 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            let data = self.0.clone().map_param(Some).into_property();
            data.name.hash(&mut hasher);
            data.target_indices.hash(&mut hasher);
            data.control_indices.hash(&mut hasher);
            data.classical_indices.hash(&mut hasher);
            for p in data.params.iter() {
                p.to_le_bytes().hash(&mut hasher);
            }
            data.pauli_ids.hash(&mut hasher);
            if let Some(matrix) = &data.unitary_matrix {
                for p in matrix.iter() {
                    for q in p.iter() {
                        q.re.to_le_bytes().hash(&mut hasher);
                        q.im.to_le_bytes().hash(&mut hasher);
                    }
                }
            }
            hasher.finish()
        }

        #[pyo3(name = "__eq__")]
        fn py_eq(&self, other: &Self) -> bool {
            self.0.clone().map_param(Some).into_property()
                == other.0.clone().map_param(Some).into_property()
        }

        #[getter]
        fn get_name(&self) -> String {
            self.0.clone().map_param(Some).into_property().name.into()
        }

        #[getter]
        fn get_target_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
            let v: Vec<usize> = slf
                .get()
                .0
                .clone()
                .map_param(Some)
                .into_property()
                .target_indices
                .into();
            PyTuple::new_bound(slf.py(), v)
        }

        #[getter]
        fn get_control_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
            let v: Vec<usize> = slf
                .get()
                .0
                .clone()
                .map_param(Some)
                .into_property()
                .control_indices
                .into();
            PyTuple::new_bound(slf.py(), v)
        }

        #[getter]
        fn get_classical_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
            let v: Vec<usize> = slf
                .get()
                .0
                .clone()
                .map_param(Some)
                .into_property()
                .classical_indices
                .into();
            PyTuple::new_bound(slf.py(), v)
        }

        #[getter]
        fn get_params<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
            let v: Vec<f64> = slf
                .get()
                .0
                .clone()
                .map_param(Some)
                .into_property()
                .params
                .into();
            PyTuple::new_bound(slf.py(), v)
        }

        #[getter]
        fn get_pauli_ids<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
            let v: Vec<u8> = slf
                .get()
                .0
                .clone()
                .map_param(Some)
                .into_property()
                .pauli_ids
                .into();
            PyTuple::new_bound(slf.py(), v)
        }

        #[getter]
        fn get_unitary_matrix<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
            if let Some(mat) = slf
                .get()
                .0
                .clone()
                .map_param(Some)
                .into_property()
                .unitary_matrix
            {
                let mat = mat
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .map(|c| {
                                // The Rust version of initialization of UnitaryMatrix gate treats all elements
                                // of the matrix as `complex` (not `float`). It may cause some
                                // backward-noncompatibility with Python codes which give `float` values
                                // to initialize UnitaryMatrix gate, because it converts `float` to `complex`
                                // unexpectedly. (An easy example is `cmath.log()`, the returning value is
                                // differ whether the input is `complex` or `float`.)
                                // To mitigate this, we convert Complex to Float when the imaginary value
                                // is near zero.
                                if c.im.abs() < 1.0e-7 {
                                    c.re.into_py(slf.py())
                                } else {
                                    c.into_py(slf.py())
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                PyTuple::new_bound(slf.py(), mat)
            } else {
                PyTuple::new_bound(slf.py(), None as Option<usize>)
            }
        }
    }

    #[pyclass(
        frozen,
        module = "quri_parts.rust.circuit.gate",
        name = "ParametricQuantumGate"
    )]
    #[derive(Clone, Debug, PartialEq)]
    struct ParametricQuantumGateWrapper(QuantumGate<()>);

    impl<T> IntoPy<T> for QuantumGate<()>
    where
        ParametricQuantumGateWrapper: IntoPy<T>,
    {
        fn into_py(self, py: Python<'_>) -> T {
            ParametricQuantumGateWrapper(self).into_py(py)
        }
    }

    impl<'py> pyo3::conversion::FromPyObject<'py> for QuantumGate<()> {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            Ok(<ParametricQuantumGateWrapper as pyo3::conversion::FromPyObject<'py>>::extract_bound(ob)?.0)
        }
    }

    #[pymethods]
    impl ParametricQuantumGateWrapper {
        #[new]
        #[pyo3(
            signature = (name, target_indices, control_indices=Vec::new(), pauli_ids=Vec::new()),
            text_signature = "(
                name: str,
                target_indices: Sequence[int],
                control_indices: Sequence[int] = [],
                pauli_ids: Sequence[int] = [],
            )"
        )]
        fn py_new(
            name: String,
            target_indices: Vec<usize>,
            control_indices: Vec<usize>,
            pauli_ids: Vec<u8>,
        ) -> PyResult<Self> {
            let prop = GenericGateProperty {
                name: name.clone().into(),
                target_indices: target_indices.into(),
                control_indices: control_indices.into(),
                pauli_ids: pauli_ids.into(),
                ..Default::default()
            };
            let mut bad_flag = false;
            let ret = prop.into_gate()?.map_param(|p| {
                if p.is_some() {
                    bad_flag = true;
                }
                ()
            });
            if !bad_flag && name.starts_with("Parametric") {
                Ok(Self(ret))
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Cannot initialize QuantumGate with {}",
                    &name
                )))
            }
        }

        #[pyo3(name = "__repr__")]
        fn py_repr(&self) -> String {
            self.0
                .clone()
                .map_param(|_| None)
                .into_property()
                .get_compat_string_parametric()
        }

        #[pyo3(name = "__reduce__")]
        fn py_reduce(
            slf: &Bound<'_, Self>,
        ) -> PyResult<(PyObject, (String, Vec<usize>, Vec<usize>, Vec<u8>))> {
            let data = &slf.get().0.clone().map_param(|_| None).into_property();
            Ok((
                slf.getattr("__class__").unwrap().unbind(),
                (
                    data.name.clone().into(),
                    data.target_indices.clone().into(),
                    data.control_indices.clone().into(),
                    data.pauli_ids.clone().into(),
                ),
            ))
        }

        #[pyo3(name = "__hash__")]
        pub(crate) fn py_hash(&self) -> u64 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            let data = self.0.clone().map_param(|_| None).into_property();
            data.name.hash(&mut hasher);
            data.target_indices.hash(&mut hasher);
            data.control_indices.hash(&mut hasher);
            for p in data.params.iter() {
                p.to_le_bytes().hash(&mut hasher);
            }
            data.pauli_ids.hash(&mut hasher);
            hasher.finish()
        }

        #[pyo3(name = "__eq__")]
        fn py_eq(&self, other: &Self) -> bool {
            self.0.clone().map_param(|_| None).into_property()
                == other.0.clone().map_param(|_| None).into_property()
        }

        #[getter]
        fn get_name(&self) -> String {
            self.0
                .clone()
                .map_param(|_| None)
                .into_property()
                .name
                .into()
        }

        #[getter]
        fn get_target_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
            let v: Vec<usize> = slf
                .get()
                .0
                .clone()
                .map_param(|_| None)
                .into_property()
                .target_indices
                .into();
            PyTuple::new_bound(slf.py(), v)
        }

        #[getter]
        fn get_control_indices<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
            let v: Vec<usize> = slf
                .get()
                .0
                .clone()
                .map_param(|_| None)
                .into_property()
                .control_indices
                .into();
            PyTuple::new_bound(slf.py(), v)
        }

        #[getter]
        fn get_pauli_ids<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
            let v: Vec<u8> = slf
                .get()
                .0
                .clone()
                .map_param(|_| None)
                .into_property()
                .pauli_ids
                .into();
            PyTuple::new_bound(slf.py(), v)
        }
    }

    pub fn add_quantum_gate(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<QuantumGateWrapper>()?;
        m.add_class::<ParametricQuantumGateWrapper>()?;
        Ok(())
    }
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "gate")?;
    wrapper::add_quantum_gate(&m)?;
    Ok(m)
}
