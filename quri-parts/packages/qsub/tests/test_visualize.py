from typing import Any

from matplotlib import patches as mpatches
from matplotlib.figure import Figure
from pytest import MonkeyPatch

import quri_parts.qsub._visualization as vis_impl
import quri_parts.qsub.visualize as visualize
from quri_parts.qsub.lib.std import (
    CNOT,
    CZ,
    SWAP,
    Cbz,
    Controlled,
    H,
    M,
    MultiControlled,
    Toffoli,
    X,
    conditional,
)
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.register import Register
from quri_parts.qsub.sub import Sub, SubBuilder
from quri_parts.qsub.visualize import (
    CircuitData,
    ControlBitInfo,
    GateData,
    GateType,
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


def _patch_drawer(monkeypatch: MonkeyPatch, record: dict[str, Any]) -> None:
    orig_init = visualize.MPLCircuitlDrawer.__init__

    def recording_init(
        self: visualize.MPLCircuitlDrawer,
        circuit_data: CircuitData,
        *,
        dpi: int,
        scale: float,
    ) -> None:
        record["data"] = circuit_data
        orig_init(self, circuit_data, dpi=dpi, scale=scale)

    monkeypatch.setattr(visualize.MPLCircuitlDrawer, "__init__", recording_init)

    class DummyBaseDrawer:
        def __init__(self, circuit_data: Any, *, dpi: int, scale: float) -> None:
            record["base_circuit"] = circuit_data
            layer_width_count = max(1, circuit_data.layer_count or 1)
            self._layer_width = [1.0] * layer_width_count
            fig = Figure()
            self._figure = fig
            self._ax = fig.add_subplot(111)

        def draw(self, **_: Any) -> Figure:
            record["drawn"] = True
            record["figure"] = self._figure
            return self._figure

    monkeypatch.setattr(vis_impl, "QVMPLCircuitlDrawer", DummyBaseDrawer)


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
        assert d == GateData("CNOT", [2], [ControlBitInfo(4, 1)])

        d = op_to_vis_data(CZ, (Qubit(4), Qubit(2)), ())
        assert d == GateData("CZ", [2], [ControlBitInfo(4, 1)])

    def test_swap_gate(self) -> None:
        d = op_to_vis_data(SWAP, (Qubit(4), Qubit(2)), ())
        assert d == GateData("SWAP", [4, 2])

    def test_toffoli_gate(self) -> None:
        d = op_to_vis_data(Toffoli, (Qubit(4), Qubit(1), Qubit(2)), ())
        assert d == GateData(
            "Toffoli", [2], [ControlBitInfo(4, 1), ControlBitInfo(1, 1)]
        )

    def test_controlled_gate(self) -> None:
        d = op_to_vis_data(Controlled(SWAP), (Qubit(4), Qubit(1), Qubit(2)), ())
        assert d == GateData("SWAP", [1, 2], [ControlBitInfo(4, 1)])

    def test_multi_controlled_gate(self) -> None:
        d = op_to_vis_data(
            MultiControlled(SWAP, 4, 5),
            tuple(Qubit(i) for i in (4, 2, 3, 5, 6, 1)),
            (),
        )
        assert d == GateData(
            "SWAP",
            [6, 1],
            [ControlBitInfo(i, c) for i, c in [(4, 1), (2, 0), (3, 1), (5, 0)]],
        )

    def test_controlled_cnot_gate(self) -> None:
        d = op_to_vis_data(Controlled(CNOT), (Qubit(4), Qubit(1), Qubit(2)), ())
        assert d == GateData("CNOT", [2], [ControlBitInfo(4, 1), ControlBitInfo(1, 1)])

    def test_multi_controlled_cnot_gate(self) -> None:
        d = op_to_vis_data(
            MultiControlled(CNOT, 4, 5),
            tuple(Qubit(i) for i in (4, 2, 3, 5, 6, 1)),
            (),
        )
        assert d == GateData(
            "CNOT",
            [1],
            [ControlBitInfo(i, c) for i, c in [(4, 1), (2, 0), (3, 1), (5, 0), (6, 1)]],
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
                ControlBitInfo(i, c)
                for i, c in [(4, 1), (2, 1), (3, 0), (5, 1), (6, 0), (1, 1)]
            ],
        )

    def test_measure_gate_includes_register_target(self) -> None:
        reg_index = {Register(0).uid: 2}
        d = op_to_vis_data(M, (Qubit(0),), (Register(0),), reg_index)
        assert d.classical_bits == [2]
        assert d.kind == GateType.MEASURE

    def test_cbz_uses_classical_control(self) -> None:
        regs = (Register(0), Register(1))
        reg_index = {regs[0].uid: 2, regs[1].uid: 3}
        d = op_to_vis_data(Cbz, (), regs, reg_index)
        assert d.target_bits == []
        assert d.classical_bits == [2]
        assert d.control_bit_infos == []
        assert d.kind == GateType.CLASSICAL


class TestSubToVisData:
    def test_classical_ops_are_included(self) -> None:
        data = sub_to_vis_data(_classical_sub())
        assert isinstance(data, CircuitData)
        assert data.qubit_count == 2
        assert data.register_count == 1
        gate_names = {gate.name for line in data.gates for gate in line}
        assert {"H", "CZ", "M", "Cbz"} <= gate_names

    def test_conditional_block_is_present(self) -> None:
        data = sub_to_vis_data(_classical_sub())
        assert len(data.conditional_blocks) == 1
        block = data.conditional_blocks[0]
        # auxiliary registers should be excluded from shaded area
        assert 2 not in block.bits
        assert set(block.bits) >= {0, 1}
        assert block.label == "r0 == 0"
        assert block.start_layer <= block.end_layer


class TestDrawSub:
    def test_handles_classical_ops(self, monkeypatch: MonkeyPatch) -> None:
        record: dict[str, Any] = {}

        _patch_drawer(monkeypatch, record)

        result = draw_sub(_classical_sub())
        assert isinstance(result, Figure)
        assert result is record["figure"]
        assert record.get("drawn") is True
        data = record["data"]
        assert data.qubit_count == 2
        assert data.register_count == 1
        base_circuit = record["base_circuit"]
        assert base_circuit.qubit_count == data.qubit_count
        assert len(base_circuit.gates) == base_circuit.qubit_count
        assert all(len(line) == base_circuit.layer_count for line in base_circuit.gates)
        gate_names = {
            gate.name
            for line in data.gates
            for gate in line
            if gate.name not in {"wire", "ghost"}
        }
        assert {"H", "CZ", "M", "Cbz"} <= gate_names
        ax = record["figure"].axes[0]
        wire_pitch = (
            vis_impl.GATE_DEFAULT_HEIGHT
            + vis_impl.GATE_MARGIN_TOP
            + vis_impl.GATE_MARGIN_BOTTOM
        )
        visual_bits = data.qubit_count + (1 if data.register_count else 0)
        expected_top = (
            visual_bits * wire_pitch
            - vis_impl.GATE_MARGIN_TOP
            - vis_impl.GATE_MARGIN_BOTTOM
            - vis_impl.GATE_DEFAULT_HEIGHT / 2
            + vis_impl.CIRCUIT_MARGIN
        )
        expected_bottom = -vis_impl.GATE_DEFAULT_HEIGHT / 2 - vis_impl.CIRCUIT_MARGIN
        ylim = ax.get_ylim()
        assert ylim[0] == expected_top
        assert ylim[1] == expected_bottom
        assert len(ax.patches) >= 2
        assert len(ax.lines) >= 3
        circles = [p for p in ax.patches if isinstance(p, mpatches.Circle)]
        cbz_circles = [
            c
            for c in circles
            if abs(c.center[1] - 2 * wire_pitch) < 1e-6 and c.get_radius() <= 0.2
        ]
        assert cbz_circles, "Cbz marker should be drawn as a filled dot"
        cond_rects = [
            p
            for p in ax.patches
            if isinstance(p, mpatches.Rectangle) and p.get_linestyle() == "--"
        ]
        assert cond_rects, "Conditional region rectangle missing"
        rect_x = cond_rects[0].get_x()
        rect_w = cond_rects[0].get_width()
        cbz_x = cbz_circles[0].center[0]
        assert rect_x <= cbz_x <= rect_x + rect_w

    def test_falls_back_to_single_op(self, monkeypatch: MonkeyPatch) -> None:
        record: dict[str, Any] = {}

        _patch_drawer(monkeypatch, record)

        result = draw_sub(CNOT)
        assert isinstance(result, Figure)
        assert result is record["figure"]
        assert record.get("drawn") is True
        assert isinstance(record["data"], CircuitData)
        base_circuit = record["base_circuit"]
        assert base_circuit.qubit_count >= 1
        assert len(base_circuit.gates) == base_circuit.qubit_count
