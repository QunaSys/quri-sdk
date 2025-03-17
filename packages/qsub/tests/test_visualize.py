from qulacsvis.models.circuit import ControlQubitInfo, GateData  # type: ignore

from quri_parts.qsub.lib.std import (
    CNOT,
    CZ,
    SWAP,
    Controlled,
    MultiControlled,
    Toffoli,
    X,
)
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.visualize import _op_controls, op_to_vis_data


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
