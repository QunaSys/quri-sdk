from typing import cast

import numpy as np
import pytest
import qulacs
from numpy.typing import NDArray

from quri_parts.circuit import QuantumCircuit, gate_names
from quri_parts.circuit.transpile.rz2hst import RZ2HSTTranspiler
from quri_parts.qulacs.circuit import convert_circuit


def test_rz2hst_generates_hstx_only() -> None:
    transpiler = RZ2HSTTranspiler(
        epsilon=1e-3, gridsynth=lambda *_: ("HSTX", float("nan"))
    )
    circuit = QuantumCircuit(1)
    circuit.add_RZ_gate(0, 0.3)
    out = transpiler(circuit)

    assert [gate.name for gate in out.gates] == [
        gate_names.H,
        gate_names.S,
        gate_names.T,
        gate_names.X,
    ]


def _rz2hst_to_qulacs_circuit(theta: float, epsilon: float) -> qulacs.QuantumCircuit:
    circuit = QuantumCircuit(1)
    circuit.add_RZ_gate(0, theta)
    transpiled = RZ2HSTTranspiler(epsilon=epsilon)(circuit)
    return convert_circuit(transpiled)


def _qulacs_circuit_to_matrix(circuit: qulacs.QuantumCircuit) -> NDArray[np.complex128]:
    # Every circuit here acts on a single qubit, so its unitary is just the
    # product of the per-gate 2x2 matrices from qulacs.QuantumGateBase.get_matrix().
    dim = 1 << circuit.get_qubit_count()
    mat: NDArray[np.complex128] = np.eye(dim, dtype=np.complex128)
    for i in range(circuit.get_gate_count()):
        gate_matrix = np.asarray(circuit.get_gate(i).get_matrix(), dtype=np.complex128)
        mat = gate_matrix @ mat
    return mat


def _align_global_phase(
    target: NDArray[np.complex128], reference: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    phase_ref = np.vdot(reference.reshape(-1), target.reshape(-1))
    if np.isclose(abs(phase_ref), 0.0):
        return target
    return cast(NDArray[np.complex128], target * np.exp(-1.0j * np.angle(phase_ref)))


@pytest.mark.gridsynth
@pytest.mark.parametrize(
    "theta",
    [
        pytest.param(0.123456, id="theta_0_123456"),
        pytest.param(-0.987654, id="theta_neg_0_987654"),
        pytest.param(float(np.pi / 7.0), id="theta_pi_over_7"),
        pytest.param(float(np.sqrt(2.0)), id="theta_sqrt2"),
        pytest.param(2.0 * np.pi * 0.37, id="theta_0_74pi"),
    ],
)
def test_rz2hst_matches_rz_with_qulacs(theta: float) -> None:
    epsilon = 1.0e-4

    rz_circuit = QuantumCircuit(1)
    rz_circuit.add_RZ_gate(0, theta)
    rz_qs = convert_circuit(rz_circuit)
    decomposed_qs = _rz2hst_to_qulacs_circuit(theta, epsilon)

    u_rz = _qulacs_circuit_to_matrix(rz_qs)
    u_decomposed = _align_global_phase(_qulacs_circuit_to_matrix(decomposed_qs), u_rz)

    assert np.allclose(u_rz, u_decomposed, atol=5.0e-4, rtol=0.0)


@pytest.mark.gridsynth
def test_rz2hst_matches_rz_exactly_with_recorded_phase() -> None:
    # With the recorded global phase applied, the decomposition matches RZ
    # exactly (no separate phase alignment) -- this pins the phase sign.
    theta = 0.123456
    rz_circuit = QuantumCircuit(1)
    rz_circuit.add_RZ_gate(0, theta)
    u_rz = _qulacs_circuit_to_matrix(convert_circuit(rz_circuit))

    transpiler = RZ2HSTTranspiler(epsilon=1.0e-6)
    decomposed = transpiler(rz_circuit)
    u_dec = _qulacs_circuit_to_matrix(convert_circuit(decomposed))

    assert np.allclose(
        u_rz, np.exp(1.0j * transpiler.phase) * u_dec, atol=1.0e-5, rtol=0.0
    )
