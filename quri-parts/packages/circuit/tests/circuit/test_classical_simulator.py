from collections import Counter
from dataclasses import dataclass
from typing import Sequence, Union

import pytest

from quri_parts.circuit import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    TOFFOLI,
    U1,
    U2,
    U3,
    H,
    Identity,
    Measurement,
    Pauli,
    PauliRotation,
    QuantumCircuit,
    QuantumGate,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    UnitaryMatrix,
    X,
    Y,
    Z,
    gate_names,
)
from quri_parts.circuit.classical_simulator import create_classical_sampler
from quri_parts.circuit.gates import (
    MCH,
    MCRX,
    MCRY,
    MCRZ,
    MCS,
    MCT,
    MCU1,
    MCX,
    MCY,
    MCZ,
    MCSdag,
    MCSqrtX,
    MCSqrtXdag,
    MCSqrtY,
    MCSqrtYdag,
    MCTdag,
)

CANVAS_QUBITS = 4
_I4 = [[1 if i == j else 0 for j in range(4)] for i in range(4)]


@dataclass(frozen=True)
class Flips:
    expected_idx: int


@dataclass(frozen=True)
class Noop:
    pass


@dataclass(frozen=True)
class Raises:
    pass


Expected = Union[Flips, Noop, Raises]


@dataclass(frozen=True)
class Case:
    gate_name: str
    case_id: str
    setup: Sequence[QuantumGate]
    gate: QuantumGate
    expected: Expected
    cbits: int = 0

    def setup_state_idx(self) -> int:
        idx = 0
        for g in self.setup:
            assert g.name == "X", "setup gates must be plain X"
            idx ^= 1 << g.target_indices[0]
        return idx


CASES: list[Case] = [
    Case("X", "X", [], X(0), Flips(0b0001)),
    Case("CNOT", "CNOT_ctrl1", [X(0)], CNOT(0, 1), Flips(0b0011)),
    Case("CNOT", "CNOT_ctrl0", [], CNOT(0, 1), Noop()),
    Case("SWAP", "SWAP", [X(0)], SWAP(0, 2), Flips(0b0100)),
    Case(
        "TOFFOLI", "TOFFOLI_both_ctrls", [X(0), X(1)], TOFFOLI(0, 1, 2), Flips(0b0111)
    ),
    Case("TOFFOLI", "TOFFOLI_one_ctrl", [X(0)], TOFFOLI(0, 1, 2), Noop()),
    Case(
        "MCX",
        "MCX_all_ctrls",
        [X(0), X(1), X(2)],
        MCX(target_index=3, control_indices=[0, 1, 2]),
        Flips(0b1111),
    ),
    Case(
        "MCX",
        "MCX_partial_ctrls",
        [X(0), X(1)],
        MCX(target_index=3, control_indices=[0, 1, 2]),
        Noop(),
    ),
    Case("Identity", "Identity", [], Identity(0), Noop()),
    Case("Z", "Z", [X(0)], Z(0), Noop()),
    Case("S", "S", [X(0)], S(0), Noop()),
    Case("Sdag", "Sdag", [X(0)], Sdag(0), Noop()),
    Case("T", "T", [X(0)], T(0), Noop()),
    Case("Tdag", "Tdag", [X(0)], Tdag(0), Noop()),
    Case("RZ", "RZ", [X(0)], RZ(0, 0.7), Noop()),
    Case("U1", "U1", [X(0)], U1(0, 0.7), Noop()),
    Case("CZ", "CZ", [X(0), X(1)], CZ(0, 1), Noop()),
    Case(
        "MCZ", "MCZ", [X(0), X(1)], MCZ(target_index=2, control_indices=[0, 1]), Noop()
    ),
    Case(
        "MCS", "MCS", [X(0), X(1)], MCS(target_index=2, control_indices=[0, 1]), Noop()
    ),
    Case(
        "MCSdag",
        "MCSdag",
        [X(0), X(1)],
        MCSdag(target_index=2, control_indices=[0, 1]),
        Noop(),
    ),
    Case(
        "MCT", "MCT", [X(0), X(1)], MCT(target_index=2, control_indices=[0, 1]), Noop()
    ),
    Case(
        "MCTdag",
        "MCTdag",
        [X(0), X(1)],
        MCTdag(target_index=2, control_indices=[0, 1]),
        Noop(),
    ),
    Case(
        "MCRZ",
        "MCRZ",
        [X(0), X(1)],
        MCRZ(target_index=2, control_indices=[0, 1], angle=0.7),
        Noop(),
    ),
    Case(
        "MCU1",
        "MCU1",
        [X(0), X(1)],
        MCU1(target_index=2, control_indices=[0, 1], angle=0.7),
        Noop(),
    ),
    Case("Pauli", "Pauli_only_I", [], Pauli([0, 1, 2], [0, 0, 0]), Noop()),
    Case("Pauli", "Pauli_I_and_X", [], Pauli([0, 1, 2], [0, 1, 1]), Flips(0b0110)),
    Case("Pauli", "Pauli_only_Z", [X(0)], Pauli([0, 1, 2], [3, 3, 3]), Noop()),
    Case("Pauli", "Pauli_X_and_Z", [X(1)], Pauli([0, 1, 2], [1, 3, 1]), Flips(0b0111)),
    Case("Pauli", "Pauli_with_Y", [], Pauli([0, 1], [1, 2]), Raises()),
    Case("Measurement", "Measurement", [X(0)], Measurement([0], [0]), Noop(), cbits=1),
    Case("Y", "Y", [], Y(0), Raises()),
    Case("H", "H", [], H(0), Raises()),
    Case("RX", "RX", [], RX(0, 0.5), Raises()),
    Case("RY", "RY", [], RY(0, 0.5), Raises()),
    Case("SqrtX", "SqrtX", [], SqrtX(0), Raises()),
    Case("SqrtXdag", "SqrtXdag", [], SqrtXdag(0), Raises()),
    Case("SqrtY", "SqrtY", [], SqrtY(0), Raises()),
    Case("SqrtYdag", "SqrtYdag", [], SqrtYdag(0), Raises()),
    Case("U2", "U2", [], U2(0, 0.1, 0.2), Raises()),
    Case("U3", "U3", [], U3(0, 0.1, 0.2, 0.3), Raises()),
    Case("MCY", "MCY", [], MCY(target_index=1, control_indices=[0]), Raises()),
    Case("MCH", "MCH", [], MCH(target_index=1, control_indices=[0]), Raises()),
    Case(
        "MCRX",
        "MCRX",
        [],
        MCRX(target_index=1, control_indices=[0], angle=0.5),
        Raises(),
    ),
    Case(
        "MCRY",
        "MCRY",
        [],
        MCRY(target_index=1, control_indices=[0], angle=0.5),
        Raises(),
    ),
    Case(
        "MCSqrtX", "MCSqrtX", [], MCSqrtX(target_index=1, control_indices=[0]), Raises()
    ),
    Case(
        "MCSqrtXdag",
        "MCSqrtXdag",
        [],
        MCSqrtXdag(target_index=1, control_indices=[0]),
        Raises(),
    ),
    Case(
        "MCSqrtY", "MCSqrtY", [], MCSqrtY(target_index=1, control_indices=[0]), Raises()
    ),
    Case(
        "MCSqrtYdag",
        "MCSqrtYdag",
        [],
        MCSqrtYdag(target_index=1, control_indices=[0]),
        Raises(),
    ),
    Case(
        "PauliRotation",
        "PauliRotation",
        [],
        PauliRotation([0, 1], [1, 1], 0.7),
        Raises(),
    ),
    Case("UnitaryMatrix", "UnitaryMatrix", [], UnitaryMatrix([0, 1], _I4), Raises()),
]


def _expected_gate_names() -> set[gate_names.GateNameType]:
    return gate_names.GATE_NAMES - gate_names.PARAMETRIC_GATE_NAMES


def test_all_gate_names_are_covered() -> None:
    assert {c.gate_name for c in CASES} == _expected_gate_names()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_gate_behaviour(case: Case) -> None:
    sampler = create_classical_sampler()
    circuit = QuantumCircuit(CANVAS_QUBITS, case.cbits)
    for g in case.setup:
        circuit.add_gate(g)
    circuit.add_gate(case.gate)

    expected = case.expected
    if isinstance(expected, Raises):
        with pytest.raises(ValueError):
            sampler(circuit, 1)
    elif isinstance(expected, Flips):
        assert sampler(circuit, 1) == Counter({expected.expected_idx: 1})
    else:
        assert sampler(circuit, 1) == Counter({case.setup_state_idx(): 1})
