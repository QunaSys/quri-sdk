from typing import Any, Optional

from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pytest import MonkeyPatch, raises

import quri_parts.qsub._visualization as vis_impl
import quri_parts.qsub.visualize as visualize
from quri_parts.qsub.eval.subwiseprofile import (
    GateCountBreakdown,
    GateCounts,
    QubitCounts,
    SubwiseProfileEntry,
)
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
from quri_parts.qsub.namespace import NameSpace
from quri_parts.qsub.op import Ident
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.register import Register
from quri_parts.qsub.sub import Sub, SubBuilder
from quri_parts.qsub.visualize import (
    CircuitData,
    ControlBitInfo,
    GateData,
    GateType,
    _call_graph_depths,
    _op_controls,
    draw_call_graph,
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

    def test_multi_controlled_swap_draws_control_markers(self) -> None:
        builder = SubBuilder(arg_qubits_count=6)
        builder.add_op(MultiControlled(SWAP, 4, 5), builder.qubits)

        figure = draw_sub(builder.build())
        try:
            control_markers = [
                patch
                for patch in figure.axes[0].patches
                if isinstance(patch, mpatches.Circle)
            ]
            control_markers.sort(key=lambda marker: marker.center[1])

            assert len(control_markers) == 4
            assert [marker.get_facecolor() for marker in control_markers] == [
                mcolors.to_rgba(color) for color in ("black", "white", "black", "white")
            ]
        finally:
            plt.close(figure)


_CG_NS = NameSpace("test_call_graph")


def _ident(local_name: str) -> Ident:
    return Ident(_CG_NS, local_name)


class TestDrawCallGraph:
    def _entry(
        self, name: str, call_count: int, cumulative: dict[str, int]
    ) -> SubwiseProfileEntry:
        counts = GateCounts(
            per_call={},
            all_calls={_ident(g): n for g, n in cumulative.items()},
        )
        empty = GateCounts(per_call={}, all_calls={})
        return SubwiseProfileEntry(
            source_op=_ident(name),
            call_count=call_count,
            qubit_counts=QubitCounts(data=0, auxiliary=0, total=0),
            gate_counts=GateCountBreakdown(
                own=empty, descendants=empty, cumulative=counts
            ),
        )

    def _texts(self, fig: Figure) -> set[str]:
        return {t.get_text() for t in fig.axes[0].texts if t.get_text()}

    def test_depths_use_longest_path(self) -> None:
        R, A, C = _ident("R"), _ident("A"), _ident("C")
        graph: dict[Optional[Ident], list[Ident]] = {R: [A, C], A: [C], C: []}
        assert _call_graph_depths(graph) == {R: 0, A: 1, C: 2}

    def test_cycle_raises(self) -> None:
        R, A, B = _ident("R"), _ident("A"), _ident("B")
        graph: dict[Optional[Ident], list[Ident]] = {R: [A], A: [B], B: [A]}
        with raises(ValueError):
            _call_graph_depths(graph)

    def test_nodes_annotated_from_profile(self) -> None:
        R, A = _ident("R"), _ident("A")
        graph: dict[Optional[Ident], list[Ident]] = {R: [A], A: []}
        profile = [
            self._entry("R", 1, {"CNOT": 6}),
            self._entry("A", 2, {"CNOT": 6}),
        ]
        fig = draw_call_graph(graph, profile)
        texts = self._texts(fig)
        plt.close(fig)
        assert "R\ncalls: 1\nCNOT: 6" in texts
        assert "A\ncalls: 2\nCNOT: 6" in texts
        assert "x2" in texts

    def test_without_profile_plain_names(self) -> None:
        R, A = _ident("R"), _ident("A")
        graph: dict[Optional[Ident], list[Ident]] = {R: [A], A: []}
        fig = draw_call_graph(graph)
        texts = self._texts(fig)
        plt.close(fig)
        assert texts == {"R", "A"}

    def test_truncated_labels_fall_back_to_qualified_names(self) -> None:
        """Distinct subs whose short names truncate to the same text are drawn
        with their fully qualified names instead of two look-alike boxes."""
        long1 = Ident(_CG_NS, "LongParam", (11111111111111111,))
        long2 = Ident(_CG_NS, "LongParam", (11111111111111112,))
        graph: dict[Optional[Ident], list[Ident]] = {long1: [], long2: []}
        fig = draw_call_graph(graph)
        texts = self._texts(fig)
        plt.close(fig)
        assert texts == {long1.to_str(full=True), long2.to_str(full=True)}

    def test_multi_caller_edge_has_no_multiplicity(self) -> None:
        R, A, B, C = _ident("R"), _ident("A"), _ident("B"), _ident("C")
        graph: dict[Optional[Ident], list[Ident]] = {
            R: [A, B],
            A: [C],
            B: [C],
            C: [],
        }
        profile = [
            self._entry("R", 1, {"X": 2}),
            self._entry("A", 1, {"X": 1}),
            self._entry("B", 1, {"X": 1}),
            self._entry("C", 2, {"X": 2}),
        ]
        fig = draw_call_graph(graph, profile)
        texts = self._texts(fig)
        plt.close(fig)
        assert "x1" in texts  # R -> A and R -> B are single-caller edges
        assert "x2" not in texts  # C has two callers: no ratio shown
