# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License.

from collections.abc import Callable
from typing import Iterable, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit.circuit.library import ECRGate, SwapGate, U2Gate, U3Gate
from qiskit.quantum_info import Operator
from qulacs import QuantumCircuit, QuantumState

from quri_parts.qiskit.circuit import circuit_from_qiskit
from quri_parts.qulacs.circuit import convert_circuit

GateCase = Tuple[str, int, Callable[[QiskitCircuit], None]]


def gate_cases() -> Iterable[GateCase]:
    return [
        ("id", 1, lambda qc: qc.id(0)),
        ("x", 1, lambda qc: qc.x(0)),
        ("y", 1, lambda qc: qc.y(0)),
        ("z", 1, lambda qc: qc.z(0)),
        ("h", 1, lambda qc: qc.h(0)),
        ("s", 1, lambda qc: qc.s(0)),
        ("sdg", 1, lambda qc: qc.sdg(0)),
        ("sx", 1, lambda qc: qc.sx(0)),
        ("sxdg", 1, lambda qc: qc.sxdg(0)),
        ("t", 1, lambda qc: qc.t(0)),
        ("tdg", 1, lambda qc: qc.tdg(0)),
        ("rx", 1, lambda qc: qc.rx(0.33, 0)),
        ("ry", 1, lambda qc: qc.ry(0.66, 0)),
        ("rz", 1, lambda qc: qc.rz(0.9, 0)),
        ("p", 1, lambda qc: qc.p(0.2, 0)),
        ("u2", 1, lambda qc: qc.append(U2Gate(0.4, 0.7), [0])),
        ("u3", 1, lambda qc: qc.append(U3Gate(0.9, 0.1, 0.2), [0])),
        ("cx", 2, lambda qc: qc.cx(0, 1)),
        ("cz", 2, lambda qc: qc.cz(0, 1)),
        ("swap", 2, lambda qc: qc.append(SwapGate(), [0, 1])),
        ("ecr", 2, lambda qc: qc.append(ECRGate(), [0, 1])),
        ("ccx", 3, lambda qc: qc.ccx(0, 1, 2)),
    ]


@pytest.mark.parametrize("name, qubit_count, builder", gate_cases())
def test_qiskit_to_qulacs_roundtrip(
    name: str, qubit_count: int, builder: Callable[[QiskitCircuit], None]
) -> None:
    pytest.importorskip("qiskit")
    qc = QiskitCircuit(qubit_count)
    builder(qc)

    qp_circuit = circuit_from_qiskit(qc, pre_conversion=False)
    qulacs_circuit: QuantumCircuit = convert_circuit(qp_circuit)
    expected = Operator(qc).data
    actual = _qulacs_circuit_to_matrix(qulacs_circuit)
    assert np.allclose(
        actual, expected, atol=1e-8, rtol=1e-6
    ), f"{name} conversion produced an unexpected unitary"


def _qulacs_circuit_to_matrix(circuit: QuantumCircuit) -> NDArray[np.complex128]:
    qubit_count = circuit.get_qubit_count()
    dim = 1 << qubit_count
    mat = np.zeros((dim, dim), dtype=np.complex128)
    state = QuantumState(qubit_count)
    for col in range(dim):
        state.set_computational_basis(col)
        circuit.update_quantum_state(state)
        mat[:, col] = state.get_vector()
    return mat
