from typing import Any

from qulacsvis.models.circuit import (  # type: ignore
    CircuitData,
    ControlQubitInfo,
    GateData,
)

import quri_parts.qsub.visualize as visualize
from quri_parts.qsub.lib.std import (
    CNOT,
    CZ,
    SWAP,
    Controlled,
    H,
    M,
    MultiControlled,
    Toffoli,
    X,
    conditional,
)
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.sub import Sub, SubBuilder
from quri_parts.qsub.visualize import (
    _op_controls,
    draw_sub,
    op_to_vis_data,
    sub_to_vis_data,
)


def _classical_sub() -> Sub:
    builder = SubBuilder(arg_qubits_count=2)
    q0, q1 = builder.qubits
    builder.add_op(H, (q0,))
    register = builder.add_aux_register()
    builder.add_op(M, (q1,), (register,))
    with conditional(builder, register):
        builder.add_op(CZ, (q0, q1))
    return builder.build()


def _patch_drawer(monkeypatch: Any, record: dict[str, Any]) -> None:
    class DummyDrawer:
        def __init__(
            self, circuit_data: CircuitData, *, dpi: int, scale: float
        ) -> None:
            record["data"] = circuit_data

        def draw(self, **_: Any) -> str:
            record["drawn"] = True
            return "figure"

    monkeypatch.setattr(visualize, "MPLCircuitlDrawer", DummyDrawer)


class TestOpControls:
    def test_cnot(self) -> None:
        c = _op_controls(CNOT, 3)
        assert list(c) == [(3, 1)]

    def test_cz(self) -> None:
        c = _op_controls(CZ, 3)
        assert list(c) == [(3, 1)]

    def test_controlled(self) -> None:
        c = _op_controls(Controlled(CNOT), 3)
        assert list(c) == [(3, 1)]

    def test_toffoli(self) -> None:
        c = _op_controls(Toffoli, 3)
        assert list(c) == [(3, 1), (4, 1)]

    def test_multi_controlled(self) -> None:
        c = _op_controls(MultiControlled(CNOT, 4, 5), 3)
        assert list(c) == [(3, 1), (4, 0), (5, 1), (6, 0)]


class TestOpToVisData:
    def test_single_qubit_gate(self) -> None:
        d = op_to_vis_data(X, (Qubit(2),), ())
        assert d == GateData("X", [2], [])

    def test_two_qubit_controlled_gate(self) -> None:
        d = op_to_vis_data(CNOT, (Qubit(4), Qubit(2)), ())
        assert d == GateData("CNOT", [2], [ControlQubitInfo(4, 1)])

        d = op_to_vis_data(CZ, (Qubit(4), Qubit(2)), ())
        assert d == GateData("CZ", [2], [ControlQubitInfo(4, 1)])

    def test_swap_gate(self) -> None:
        d = op_to_vis_data(SWAP, (Qubit(4), Qubit(2)), ())
        assert d == GateData("SWAP", [4, 2])

    def test_toffoli_gate(self) -> None:
        d = op_to_vis_data(Toffoli, (Qubit(4), Qubit(1), Qubit(2)), ())
        assert d == GateData(
            "Toffoli", [2], [ControlQubitInfo(4, 1), ControlQubitInfo(1, 1)]
        )

    def test_controlled_gate(self) -> None:
        d = op_to_vis_data(Controlled(SWAP), (Qubit(4), Qubit(1), Qubit(2)), ())
        assert d == GateData("SWAP", [1, 2], [ControlQubitInfo(4, 1)])

    def test_multi_controlled_gate(self) -> None:
        d = op_to_vis_data(
            MultiControlled(SWAP, 4, 5),
            tuple(Qubit(i) for i in (4, 2, 3, 5, 6, 1)),
            (),
        )
        assert d == GateData(
            "SWAP",
            [6, 1],
            [ControlQubitInfo(i, c) for i, c in [(4, 1), (2, 0), (3, 1), (5, 0)]],
        )

    def test_controlled_cnot_gate(self) -> None:
        d = op_to_vis_data(Controlled(CNOT), (Qubit(4), Qubit(1), Qubit(2)), ())
        assert d == GateData(
            "CNOT", [2], [ControlQubitInfo(4, 1), ControlQubitInfo(1, 1)]
        )

    def test_multi_controlled_cnot_gate(self) -> None:
        d = op_to_vis_data(
            MultiControlled(CNOT, 4, 5),
            tuple(Qubit(i) for i in (4, 2, 3, 5, 6, 1)),
            (),
        )
        assert d == GateData(
            "CNOT",
            [1],
            [
                ControlQubitInfo(i, c)
                for i, c in [(4, 1), (2, 0), (3, 1), (5, 0), (6, 1)]
            ],
        )

    def test_nested_controlled_gate(self) -> None:
        d = op_to_vis_data(
            Controlled(MultiControlled(Controlled(X), 4, 5)),
            tuple(Qubit(i) for i in (4, 2, 3, 5, 6, 1, 0)),
            (),
        )
        assert d == GateData(
            "X",
            [0],
            [
                ControlQubitInfo(i, c)
                for i, c in [(4, 1), (2, 1), (3, 0), (5, 1), (6, 0), (1, 1)]
            ],
        )


class TestSubToVisData:
    def test_classical_ops_are_ignored(self) -> None:
        data = sub_to_vis_data(_classical_sub())
        assert isinstance(data, CircuitData)
        assert data.qubit_count == 2
        gate_names = {gate.name for line in data.gates for gate in line}
        assert {"H", "CZ"} <= gate_names


class TestDrawSub:
    def test_handles_classical_ops(self, monkeypatch) -> None:
        record: dict[str, Any] = {}

        _patch_drawer(monkeypatch, record)

        result = draw_sub(_classical_sub())
        assert result == "figure"
        assert record.get("drawn") is True
        data = record["data"]
        assert data.qubit_count == 2
        gate_names = {
            gate.name
            for line in data.gates
            for gate in line
            if gate.name not in {"wire", "ghost"}
        }
        assert {"H", "CZ"} <= gate_names

    def test_falls_back_to_single_op(self, monkeypatch) -> None:
        record: dict[str, Any] = {}

        _patch_drawer(monkeypatch, record)

        result = draw_sub(CNOT)
        assert result == "figure"
        assert record.get("drawn") is True
        assert isinstance(record["data"], CircuitData)
