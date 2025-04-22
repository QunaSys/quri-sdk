use crate::circuit::gate::{ParametricQuantumGate, QuantumGate};
use crate::circuit::parameter::Parameter;
use crate::circuit::MaybeUnbound;
use num_complex::{Complex64, ComplexFloat};
use pyo3::prelude::*;

#[pyfunction(
    name = "Identity",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn identity(target_index: usize) -> QuantumGate {
    QuantumGate::Identity(target_index)
}

#[pyfunction(
    name = "X",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn x(target_index: usize) -> QuantumGate {
    QuantumGate::X(target_index)
}

#[pyfunction(
    name = "Y",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn y(target_index: usize) -> QuantumGate {
    QuantumGate::Y(target_index)
}

#[pyfunction(
    name = "Z",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn z(target_index: usize) -> QuantumGate {
    QuantumGate::Z(target_index)
}

#[pyfunction(
    name = "H",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn h(target_index: usize) -> QuantumGate {
    QuantumGate::H(target_index)
}

#[pyfunction(
    name = "S",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn s(target_index: usize) -> QuantumGate {
    QuantumGate::S(target_index)
}

#[pyfunction(
    name = "Sdag",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn sdag(target_index: usize) -> QuantumGate {
    QuantumGate::Sdag(target_index)
}

#[pyfunction(
    name = "SqrtX",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn sqrtx(target_index: usize) -> QuantumGate {
    QuantumGate::SqrtX(target_index)
}

#[pyfunction(
    name = "SqrtXdag",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn sqrtxdag(target_index: usize) -> QuantumGate {
    QuantumGate::SqrtXdag(target_index)
}

#[pyfunction(
    name = "SqrtY",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn sqrty(target_index: usize) -> QuantumGate {
    QuantumGate::SqrtY(target_index)
}

#[pyfunction(
    name = "SqrtYdag",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn sqrtydag(target_index: usize) -> QuantumGate {
    QuantumGate::SqrtYdag(target_index)
}

#[pyfunction(
    name = "T",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn t(target_index: usize) -> QuantumGate {
    QuantumGate::T(target_index)
}

#[pyfunction(
    name = "Tdag",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn tdag(target_index: usize) -> QuantumGate {
    QuantumGate::Tdag(target_index)
}

#[pyfunction(
    name = "RX",
    signature = (target_index, angle),
    text_signature = "(target_index: int, angle: float)",
)]
pub fn rx<'py>(target_index: usize, angle: f64) -> QuantumGate {
    QuantumGate::RX(target_index, angle)
}

#[pyfunction(
    name = "RY",
    signature = (target_index, angle),
    text_signature = "(target_index: int, angle: float)",
)]
pub fn ry<'py>(target_index: usize, angle: f64) -> QuantumGate {
    QuantumGate::RY(target_index, angle)
}

#[pyfunction(
    name = "RZ",
    signature = (target_index, angle),
    text_signature = "(target_index: int, angle: float)",
)]
pub fn rz<'py>(target_index: usize, angle: f64) -> QuantumGate {
    QuantumGate::RZ(target_index, angle)
}

#[pyfunction(
    name = "ParametricRX",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn parametric_rx(target_index: usize) -> ParametricQuantumGate {
    QuantumGate::RX(target_index, ())
}

#[pyfunction(
    name = "ParametricRY",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn parametric_ry(target_index: usize) -> ParametricQuantumGate {
    QuantumGate::RY(target_index, ())
}

#[pyfunction(
    name = "ParametricRZ",
    signature = (target_index),
    text_signature = "(target_index: int)",
)]
pub fn parametric_rz(target_index: usize) -> ParametricQuantumGate {
    QuantumGate::RZ(target_index, ())
}

#[pyfunction(
    name = "ParametricPauliRotation",
    signature = (target_indices, pauli_ids),
    text_signature = "(target_index: Sequence[int], pauli_ids: Sequence[int])",
)]
pub fn parametric_pauli_rotation(
    target_indices: Vec<usize>,
    pauli_ids: Vec<u8>,
) -> ParametricQuantumGate {
    QuantumGate::PauliRotation(target_indices, pauli_ids, ())
}

#[pyfunction(
    name = "U1",
    signature = (target_index, lmd),
    text_signature = "(target_index: int, lmd: float)",
)]
pub fn u1<'py>(target_index: usize, lmd: f64) -> QuantumGate {
    QuantumGate::U1(target_index, lmd)
}

#[pyfunction(
    name = "U2",
    signature = (target_index, phi, lmd),
    text_signature = "(target_index: int, phi: float, lmd: float)",
)]
pub fn u2<'py>(target_index: usize, phi: f64, lmd: f64) -> QuantumGate {
    QuantumGate::U2(target_index, phi, lmd)
}

#[pyfunction(
    name = "U3",
    signature = (target_index, theta, phi, lmd),
    text_signature = "(target_index: int, theta: float, phi: float, lmd: float)",
)]
pub fn u3<'py>(target_index: usize, theta: f64, phi: f64, lmd: f64) -> QuantumGate {
    QuantumGate::U3(target_index, theta, phi, lmd)
}

#[pyfunction(
    name = "CNOT",
    signature = (control_index, target_index),
    text_signature = "(control_index: int, target_index: int)",
)]
pub fn cnot(control_index: usize, target_index: usize) -> QuantumGate {
    QuantumGate::CNOT(control_index, target_index)
}

#[pyfunction(
    name = "CZ",
    signature = (control_index, target_index),
    text_signature = "(control_index: int, target_index: int)",
)]
pub fn cz(control_index: usize, target_index: usize) -> QuantumGate {
    QuantumGate::CZ(control_index, target_index)
}

#[pyfunction(
    name = "SWAP",
    signature = (target_index1, target_index2),
    text_signature = "(target_index1: int, target_index2: int)",
)]
pub fn swap(target_index1: usize, target_index2: usize) -> QuantumGate {
    QuantumGate::SWAP(target_index1, target_index2)
}

#[pyfunction(
    name = "TOFFOLI",
    signature = (control_index1, control_index2, target_index),
    text_signature = "(control_index1: int, control_index2: int, target_index: int)",
)]
pub fn toffoli(control_index1: usize, control_index2: usize, target_index: usize) -> QuantumGate {
    QuantumGate::TOFFOLI(control_index1, control_index2, target_index)
}

#[pyfunction(
    name = "UnitaryMatrix",
    signature = (target_indices, unitary_matrix),
    text_signature = "(target_indices: Sequence[int], unitary_matrix: Sequence[Sequence[complex]])",
)]
pub fn unitary_matrix(
    target_indices: Vec<usize>,
    unitary_matrix: Vec<Vec<Complex64>>,
) -> PyResult<QuantumGate> {
    let dim = 2usize.pow(target_indices.len() as u32);
    if unitary_matrix.len() == dim && unitary_matrix.iter().all(|v| v.len() == dim) {
        if (0..dim).all(|i| {
            (0..dim).all(|j| {
                ((0..dim)
                    .map(|k| unitary_matrix[i][k] * unitary_matrix[j][k].conj())
                    .sum::<Complex64>()
                    - Complex64::new(Into::<u8>::into(i == j) as f64, 0f64))
                .abs()
                    < 1e-5
            })
        }) {
            Ok(QuantumGate::UnitaryMatrix(
                target_indices.into(),
                unitary_matrix.into_iter().map(Into::into).collect(),
            ))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "The given matrix is not unitary.",
            ))
        }
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "The number of qubits does not match the size of the unitary matrix.",
        ))
    }
}

#[pyfunction(
    name = "SingleQubitUnitaryMatrix",
    signature = (target_index, mat),
    text_signature = "(target_index: int, unitary_matrix: Sequence[Sequence[complex]])",
)]
pub fn single_qubit_unitary_matrix(
    target_index: usize,
    mat: Vec<Vec<Complex64>>,
) -> PyResult<QuantumGate> {
    unitary_matrix(vec![target_index], mat)
}

#[pyfunction(
    name = "TwoQubitUnitaryMatrix",
    signature = (target_index1, target_index2, mat),
    text_signature = "(target_index1: int, target_index2: int, unitary_matrix: Sequence[Sequence[complex]])",
)]
pub fn two_qubit_unitary_matrix(
    target_index1: usize,
    target_index2: usize,
    mat: Vec<Vec<Complex64>>,
) -> PyResult<QuantumGate> {
    unitary_matrix(vec![target_index1, target_index2], mat)
}

#[pyfunction(
    name = "Pauli",
    signature = (target_indices, pauli_ids),
    text_signature = "(target_indices: Sequence[int], pauli_ids: Sequence[int])",
)]
pub fn pauli(target_indices: Vec<usize>, pauli_ids: Vec<u8>) -> QuantumGate {
    QuantumGate::Pauli(target_indices.into(), pauli_ids.into())
}

#[pyfunction(
    name = "PauliRotation",
    signature = (target_indices, pauli_ids, angle),
    text_signature = "(target_indices: Sequence[int], pauli_ids: Sequence[int], angle: float)",
)]
pub fn pauli_rotation(target_indices: Vec<usize>, pauli_ids: Vec<u8>, angle: f64) -> QuantumGate {
    QuantumGate::PauliRotation(target_indices.into(), pauli_ids.into(), angle)
}

#[pyfunction(
    name = "Measurement",
    signature = (target_indices, classical_indices),
    text_signature = "(target_indices: Sequence[int], classical_indices: Sequence[int])",
)]
pub fn measurement(
    target_indices: Vec<usize>,
    classical_indices: Vec<usize>,
) -> PyResult<QuantumGate> {
    if target_indices.len() != classical_indices.len() {
        Err(pyo3::exceptions::PyValueError::new_err(
            "Number of qubits and classical bits must be same for measurement.",
        ))
    } else {
        Ok(QuantumGate::Measurement(
            target_indices.into(),
            classical_indices.into(),
        ))
    }
}

pub fn py_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new_bound(py, "gates")?;
    m.add_wrapped(wrap_pyfunction!(identity))?;
    m.add_wrapped(wrap_pyfunction!(x))?;
    m.add_wrapped(wrap_pyfunction!(y))?;
    m.add_wrapped(wrap_pyfunction!(z))?;
    m.add_wrapped(wrap_pyfunction!(h))?;
    m.add_wrapped(wrap_pyfunction!(s))?;
    m.add_wrapped(wrap_pyfunction!(sdag))?;
    m.add_wrapped(wrap_pyfunction!(sqrtx))?;
    m.add_wrapped(wrap_pyfunction!(sqrtxdag))?;
    m.add_wrapped(wrap_pyfunction!(sqrty))?;
    m.add_wrapped(wrap_pyfunction!(sqrtydag))?;
    m.add_wrapped(wrap_pyfunction!(t))?;
    m.add_wrapped(wrap_pyfunction!(tdag))?;
    m.add_wrapped(wrap_pyfunction!(rx))?;
    m.add_wrapped(wrap_pyfunction!(ry))?;
    m.add_wrapped(wrap_pyfunction!(rz))?;
    m.add_wrapped(wrap_pyfunction!(parametric_rx))?;
    m.add_wrapped(wrap_pyfunction!(parametric_ry))?;
    m.add_wrapped(wrap_pyfunction!(parametric_rz))?;
    m.add_wrapped(wrap_pyfunction!(parametric_pauli_rotation))?;
    m.add_wrapped(wrap_pyfunction!(u1))?;
    m.add_wrapped(wrap_pyfunction!(u2))?;
    m.add_wrapped(wrap_pyfunction!(u3))?;
    m.add_wrapped(wrap_pyfunction!(cnot))?;
    m.add_wrapped(wrap_pyfunction!(cz))?;
    m.add_wrapped(wrap_pyfunction!(swap))?;
    m.add_wrapped(wrap_pyfunction!(toffoli))?;
    m.add_wrapped(wrap_pyfunction!(unitary_matrix))?;
    m.add_wrapped(wrap_pyfunction!(single_qubit_unitary_matrix))?;
    m.add_wrapped(wrap_pyfunction!(two_qubit_unitary_matrix))?;
    m.add_wrapped(wrap_pyfunction!(pauli))?;
    m.add_wrapped(wrap_pyfunction!(pauli_rotation))?;
    m.add_wrapped(wrap_pyfunction!(measurement))?;
    Ok(m)
}

impl QuantumGate<MaybeUnbound> {
    pub fn instantiate<'py>(
        self,
    ) -> PyResult<Result<QuantumGate, (ParametricQuantumGate, Parameter)>> {
        match self {
            Self::Identity(q) => Ok(Ok(identity(q))),
            Self::X(q) => Ok(Ok(x(q))),
            Self::Y(q) => Ok(Ok(y(q))),
            Self::Z(q) => Ok(Ok(z(q))),
            Self::H(q) => Ok(Ok(h(q))),
            Self::S(q) => Ok(Ok(s(q))),
            Self::Sdag(q) => Ok(Ok(sdag(q))),
            Self::SqrtX(q) => Ok(Ok(sqrtx(q))),
            Self::SqrtXdag(q) => Ok(Ok(sqrtxdag(q))),
            Self::SqrtY(q) => Ok(Ok(sqrty(q))),
            Self::SqrtYdag(q) => Ok(Ok(sqrtydag(q))),
            Self::T(q) => Ok(Ok(t(q))),
            Self::Tdag(q) => Ok(Ok(tdag(q))),
            Self::CNOT(q0, q1) => Ok(Ok(cnot(q0, q1))),
            Self::CZ(q0, q1) => Ok(Ok(cz(q0, q1))),
            Self::SWAP(q0, q1) => Ok(Ok(swap(q0, q1))),
            Self::TOFFOLI(q0, q1, q2) => Ok(Ok(toffoli(q0, q1, q2))),
            Self::UnitaryMatrix(qs, mat) => Ok(Ok(unitary_matrix(
                qs.into(),
                mat.into_iter().map(Into::into).collect(),
            )?)),
            Self::RX(q, p) => match p {
                MaybeUnbound::Bound(p) => Ok(Ok(rx(q, p))),
                MaybeUnbound::Unbound(pid) => Ok(Err((parametric_rx(q), pid))),
            },
            Self::RY(q, p) => match p {
                MaybeUnbound::Bound(p) => Ok(Ok(ry(q, p))),
                MaybeUnbound::Unbound(pid) => Ok(Err((parametric_ry(q), pid))),
            },
            Self::RZ(q, p) => match p {
                MaybeUnbound::Bound(p) => Ok(Ok(rz(q, p))),
                MaybeUnbound::Unbound(pid) => Ok(Err((parametric_rz(q), pid))),
            },
            Self::U1(q, p) => Ok(Ok(u1(q, p))),
            Self::U2(q, p0, p1) => Ok(Ok(u2(q, p0, p1))),
            Self::U3(q, p0, p1, p2) => Ok(Ok(u3(q, p0, p1, p2))),
            Self::Pauli(qs, ps) => Ok(Ok(pauli(qs.into(), ps.into()))),
            Self::PauliRotation(qs, ps, a) => match a {
                MaybeUnbound::Bound(a) => Ok(Ok(pauli_rotation(qs.into(), ps.into(), a))),
                MaybeUnbound::Unbound(pid) => {
                    Ok(Err((parametric_pauli_rotation(qs.into(), ps.into()), pid)))
                }
            },
            Self::Measurement(qs, cs) => Ok(Ok(measurement(qs.into(), cs.into())?)),
            Self::Other(o) => Ok(Ok(QuantumGate::Other(o))),
        }
    }
}
