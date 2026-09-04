# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import deque

# mypy: disable-error-code=import-untyped
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
from math import hypot
from typing import Any, Deque, List, Optional, Sequence, Tuple, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from qulacsvis.models.circuit import CircuitData as QVCircuitData
from qulacsvis.models.circuit import ControlQubitInfo as QVControlQubitInfo
from qulacsvis.models.circuit import GateData as QVGateData
from qulacsvis.utils import gate as qv_gate
from qulacsvis.visualization import matplotlib as qv_mpl
from qulacsvis.visualization.matplotlib import MPLCircuitlDrawer as QVMPLCircuitlDrawer
from typing_extensions import Final

grouping_adjacent_gates = qv_gate.grouping_adjacent_gates
to_latex_style = qv_gate.to_latex_style

# Visualization models ---------------------------------------------------------


@dataclass
class ControlBitInfo:
    index: int
    control_value: int


class GateType(Enum):
    GENERIC = "generic"
    MEASURE = "measure"
    CLASSICAL = "classical"


@dataclass
class GateData:
    name: str
    target_bits: List[int] = field(default_factory=list)
    control_bit_infos: List[ControlBitInfo] = field(default_factory=list)
    classical_bits: List[int] = field(default_factory=list)
    kind: GateType = GateType.GENERIC

    @property
    def all_target_bits(self) -> Tuple[int, ...]:
        return tuple(chain(self.target_bits, self.classical_bits))

    @property
    def indices(self) -> Tuple[int, ...]:
        return tuple(
            chain(self.all_target_bits, (c.index for c in self.control_bit_infos))
        )

    @property
    def max_index(self) -> int:
        return max(self.indices)

    @property
    def min_index(self) -> int:
        return min(self.indices)


GateDataSeq = Sequence[Sequence[GateData]]


@dataclass
class ConditionalBlock:
    start_layer: int
    end_layer: int
    bits: List[int]
    label: str
    cbz_layer: Optional[int] = None
    cbz_bit: Optional[int] = None


@dataclass
class CircuitData:
    qubit_count: int
    register_count: int
    layer_count: int
    gates: GateDataSeq
    conditional_blocks: List[ConditionalBlock] = field(default_factory=list)

    @property
    def bit_count(self) -> int:
        return self.qubit_count + self.register_count

    @staticmethod
    def from_gate_sequence(
        gates: Sequence[GateData],
        *,
        qubit_count: Optional[int] = None,
        register_count: int = 0,
        conditional_blocks: Optional[Sequence[ConditionalBlock]] = None,
    ) -> "CircuitData":
        if qubit_count is None:
            max_index = max((g.max_index for g in gates), default=-1)
            qubit_count = max_index + 1 - register_count if max_index >= 0 else 0
        bit_count = qubit_count + register_count
        if bit_count <= 0:
            bit_count = 1

        temp_lines: List[Deque[GateData]] = [deque() for _ in range(bit_count)]
        for gate in gates:
            targets = gate.all_target_bits
            if not targets:
                continue
            primary_index = targets[0]
            _align_layers(temp_lines, gate.min_index, gate.max_index)

            for index in range(gate.min_index, gate.max_index + 1):
                line = temp_lines[index]
                if index == primary_index:
                    line.append(gate)
                elif index in targets:
                    line.append(GateData("ghost"))
                else:
                    line.append(GateData("wire"))

        _align_layers(temp_lines, 0, bit_count - 1)
        layer_count = len(temp_lines[0])
        return CircuitData(
            qubit_count=qubit_count,
            register_count=register_count,
            layer_count=layer_count,
            gates=[list(queue) for queue in temp_lines],
            conditional_blocks=list(conditional_blocks or []),
        )


def _align_layers(
    lines: Sequence[Deque[GateData]], min_line_index: int, max_line_index: int
) -> None:
    if min_line_index > max_line_index:
        min_line_index, max_line_index = max_line_index, min_line_index
    max_line_index += 1
    lines = lines[min_line_index:max_line_index]
    layer_counts = [len(queue) for queue in lines]
    max_layer_count = max(layer_counts) if layer_counts else 0

    for queue, layer_count in zip(lines, layer_counts):
        for _ in range(max_layer_count - layer_count):
            queue.append(GateData("wire"))


# Matplotlib drawer -----------------------------------------------------------

MATPLOTLIB_INLINE_BACKENDS = qv_mpl.MATPLOTLIB_INLINE_BACKENDS

GATE_DEFAULT_WIDTH: float = cast(float, qv_mpl.GATE_DEFAULT_WIDTH)
GATE_DEFAULT_HEIGHT: float = cast(float, qv_mpl.GATE_DEFAULT_HEIGHT)

GATE_MARGIN_RIGHT: Final[float] = cast(float, qv_mpl.GATE_MARGIN_RIGHT)
GATE_MARGIN_LEFT: Final[float] = cast(float, qv_mpl.GATE_MARGIN_LEFT)
GATE_MARGIN_BOTTOM: Final[float] = cast(float, qv_mpl.GATE_MARGIN_BOTTOM)
GATE_MARGIN_TOP: Final[float] = cast(float, qv_mpl.GATE_MARGIN_TOP)

CIRCUIT_MARGIN: float = cast(float, qv_mpl.CIRCUIT_MARGIN)

PORDER_LINE: Final[int] = cast(int, qv_mpl.PORDER_LINE)
PORDER_GATE: Final[int] = cast(int, qv_mpl.PORDER_GATE)
PORDER_TEXT: Final[int] = cast(int, qv_mpl.PORDER_TEXT)


def _calc_gate_width(gate: GateData) -> float:
    width: float = GATE_DEFAULT_WIDTH
    if gate.name == "":
        return float(width)
    try:
        to_latex_style(gate.name)
    except KeyError:
        char_width = 0.2
        max_line_width = max(len(s) for s in gate.name.splitlines())
        width += (max_line_width - 3) * char_width

    return width


def _wire_pitch() -> float:
    return GATE_DEFAULT_HEIGHT + GATE_MARGIN_TOP + GATE_MARGIN_BOTTOM


def _calc_layer_positions(layer_widths: Sequence[float]) -> List[float]:
    if not layer_widths:
        return [0.0]

    layer_positions: List[float] = []
    xpos = 0.0
    for layer, width in enumerate(layer_widths):
        layer_positions.append(xpos)
        if layer < len(layer_widths) - 1:
            xpos += (
                (width + layer_widths[layer + 1]) * 0.5
                + GATE_MARGIN_RIGHT
                + GATE_MARGIN_LEFT
            )

    return layer_positions


def _circuit_span(
    layer_widths: Sequence[float], circuit_layer_count: int
) -> Tuple[float, float]:
    if not layer_widths:
        layer_widths = [GATE_DEFAULT_WIDTH]

    circuit_max_x = (
        sum(layer_widths)
        + circuit_layer_count * (GATE_MARGIN_RIGHT + GATE_MARGIN_LEFT)
        - layer_widths[0] / 2
    )
    circuit_min_x = -layer_widths[0] / 2 - GATE_MARGIN_LEFT - GATE_MARGIN_RIGHT
    return circuit_min_x, circuit_max_x


def _to_qulacsvis_gate(gate: GateData, qubit_count: int) -> Any:
    if gate.name in {"ghost", "wire"}:
        return QVGateData(gate.name)

    qubit_targets = [bit for bit in gate.target_bits if bit < qubit_count]
    controls = [
        QVControlQubitInfo(info.index, info.control_value)
        for info in gate.control_bit_infos
        if info.index < qubit_count
    ]
    return QVGateData(gate.name, qubit_targets, controls)


def _to_qulacsvis_circuit(circuit: CircuitData) -> Any:
    qubit_count = max(1, circuit.qubit_count)
    gates: List[List[Any]] = [[] for _ in range(qubit_count)]
    if circuit.layer_count == 0:
        return QVCircuitData(
            qubit_count=qubit_count,
            layer_count=0,
            gates=gates,
        )

    for layer in range(circuit.layer_count):
        for qubit in range(qubit_count):
            if qubit < circuit.qubit_count:
                gate = circuit.gates[qubit][layer]
            else:
                gate = GateData("wire")
            gates[qubit].append(_to_qulacsvis_gate(gate, circuit.qubit_count))

    return QVCircuitData(
        qubit_count=qubit_count,
        layer_count=circuit.layer_count,
        gates=gates,
    )


def _patch_swap_controls(drawer: Any) -> None:
    """qulacsvis' drawer doesn't draw control markers for SWAP gates."""
    original_swap = getattr(drawer, "_swap", None)
    if original_swap is None:
        return

    def _swap(gate: Any, xy: Tuple[float, float]) -> None:
        original_swap(gate, xy)
        if gate.control_bit_infos:
            drawer._control_bits(gate.control_bit_infos, xy)

    drawer._swap = _swap


class MPLCircuitlDrawer:
    def __init__(
        self,
        circuit: CircuitData,
        *,
        dpi: int = 72,
        scale: float = 0.6,
    ):
        self._circuit = circuit
        self._dpi = dpi
        self._fig_scale_factor = scale
        self._wire_pitch = _wire_pitch()
        self._layer_width: List[float] = []
        self._layer_positions: List[float] = []
        self._visual_bit_count: int = circuit.qubit_count + (
            1 if circuit.register_count > 0 else 0
        )
        self._circuit_min_x: float = 0.0
        self._circuit_max_x: float = 0.0
        self._figure: matplotlib.figure.Figure | None = None
        self._ax: matplotlib.axes.Axes | None = None

    def _bit_ypos(self, bit_index: int) -> float:
        visual_index = (
            bit_index
            if bit_index < self._circuit.qubit_count
            else self._circuit.qubit_count
        )
        return visual_index * self._wire_pitch

    def draw(
        self, *, debug: bool = False, filename: Optional[str] = None
    ) -> matplotlib.figure.Figure:
        qv_circuit = _to_qulacsvis_circuit(self._circuit)
        base_drawer = QVMPLCircuitlDrawer(
            qv_circuit, dpi=self._dpi, scale=self._fig_scale_factor
        )
        _patch_swap_controls(base_drawer)
        figure: matplotlib.figure.Figure = base_drawer.draw(debug=debug, filename=None)

        self._figure = figure
        self._ax = base_drawer._ax
        self._layer_width = list(base_drawer._layer_width)
        self._layer_positions = _calc_layer_positions(self._layer_width)
        self._circuit_min_x, self._circuit_max_x = _circuit_span(
            self._layer_width, qv_circuit.layer_count
        )

        self._extend_limits()
        self._draw_conditional_blocks()
        self._draw_register_lines()
        self._draw_measurement_arrows()
        self._draw_classical_gates()
        self._draw_classical_controls()

        if filename:
            figure.savefig(filename)

        if matplotlib.get_backend() in MATPLOTLIB_INLINE_BACKENDS:
            plt.close(figure)
        return figure

    def _extend_limits(self) -> None:
        if self._ax is None or self._figure is None:
            return

        circuit_max_y = (
            self._visual_bit_count * self._wire_pitch
            - GATE_MARGIN_TOP
            - GATE_MARGIN_BOTTOM
            - GATE_DEFAULT_HEIGHT / 2
        )
        qubit_label_width = 2
        self._ax.set_xlim(
            self._circuit_min_x - qubit_label_width,
            self._circuit_max_x + CIRCUIT_MARGIN,
        )
        self._ax.set_ylim(
            circuit_max_y + CIRCUIT_MARGIN,
            -GATE_DEFAULT_HEIGHT / 2 - CIRCUIT_MARGIN,
        )

        fig_width = abs(self._ax.get_xlim()[1] - self._ax.get_xlim()[0])
        fig_heigth = abs(self._ax.get_ylim()[1] - self._ax.get_ylim()[0])
        self._figure.set_size_inches(
            fig_width * self._fig_scale_factor, fig_heigth * self._fig_scale_factor
        )

    def _draw_register_lines(self) -> None:
        if self._ax is None:
            return

        if self._circuit.register_count:
            bit_index = self._circuit.qubit_count
            line_ypos = self._bit_ypos(bit_index)
            label = (
                f"r[0..{self._circuit.register_count - 1}]"
                if self._circuit.register_count > 1
                else "r"
            )
            self._text(self._circuit_min_x - 1, line_ypos, label, fontsize=24)
            self._double_line(
                (self._circuit_min_x, line_ypos),
                (self._circuit_max_x, line_ypos),
                lw=1.8,
            )

    def _draw_measurement_arrows(self) -> None:
        if self._ax is None or not self._layer_positions:
            return

        for layer in range(self._circuit.layer_count):
            xpos = self._layer_positions[min(layer, len(self._layer_positions) - 1)]
            for bit in range(min(self._circuit.qubit_count, len(self._circuit.gates))):
                gate = self._circuit.gates[bit][layer]
                if gate.kind is not GateType.MEASURE or not gate.classical_bits:
                    continue
                ypos = self._bit_ypos(bit)
                start_y = ypos + GATE_DEFAULT_HEIGHT * 0.5
                self._double_line((xpos, ypos), (xpos, start_y), lw=1.6)
                for i, target in enumerate(gate.classical_bits):
                    to_ypos = self._bit_ypos(target)
                    offset_x = xpos + i * 0.12
                    self._ax.annotate(
                        "",
                        xy=(offset_x, to_ypos),
                        xytext=(offset_x, start_y),
                        arrowprops={"arrowstyle": "->", "linewidth": 2.0},
                        zorder=PORDER_GATE,
                    )
                    self._text(
                        offset_x - 0.15,
                        to_ypos - 0.4,
                        f"r{target - self._circuit.qubit_count}",
                        fontsize=20,
                        horizontalalignment="right",
                        verticalalignment="top",
                    )

    def _draw_classical_gates(self) -> None:
        if self._ax is None or not self._layer_positions:
            return

        for layer in range(self._circuit.layer_count):
            xpos = self._layer_positions[min(layer, len(self._layer_positions) - 1)]
            for bit in range(self._circuit.qubit_count, self._circuit.bit_count):
                gate = self._circuit.gates[bit][layer]
                if (
                    gate.name in {"wire", "ghost", "Cbz"}
                    or gate.kind is not GateType.CLASSICAL
                ):
                    continue
                ypos = self._bit_ypos(bit)
                if len(gate.all_target_bits) > 1:
                    self._multi_gate(gate, (xpos, ypos))
                else:
                    self._gate_with_size(gate, (xpos, ypos), 1)

    def _draw_classical_controls(self) -> None:
        if self._ax is None or not self._layer_positions:
            return

        for layer in range(self._circuit.layer_count):
            xpos = self._layer_positions[min(layer, len(self._layer_positions) - 1)]
            for bit in range(self._circuit.bit_count):
                gate = self._circuit.gates[bit][layer]
                if gate.name in {"wire", "ghost"}:
                    continue
                self._control_bits(gate.control_bit_infos, (xpos, self._bit_ypos(bit)))

    def _draw_conditional_blocks(self) -> None:
        if (
            self._ax is None
            or not self._circuit.conditional_blocks
            or not self._layer_positions
        ):
            return

        for block in self._circuit.conditional_blocks:
            start_x = (
                self._layer_positions[block.start_layer]
                - self._layer_width[block.start_layer] * 0.5
                - GATE_MARGIN_LEFT
            )
            end_x = (
                self._layer_positions[block.end_layer]
                + self._layer_width[block.end_layer] * 0.5
                + GATE_MARGIN_RIGHT
            )
            min_bit = min(block.bits)
            max_bit = max(block.bits)
            y0 = self._bit_ypos(min_bit) - (
                GATE_DEFAULT_HEIGHT / 2 + GATE_MARGIN_BOTTOM
            )
            y1 = self._bit_ypos(max_bit) + (GATE_DEFAULT_HEIGHT / 2 + GATE_MARGIN_TOP)

            rect = patches.Rectangle(
                (start_x, y0),
                end_x - start_x,
                y1 - y0,
                facecolor="#f5f5f5",
                edgecolor="#999999",
                linewidth=1.5,
                linestyle="--",
                zorder=PORDER_LINE - 1,
                alpha=0.8,
            )
            self._ax.add_patch(rect)
            if block.cbz_layer is not None and block.cbz_bit is not None:
                marker_x = self._layer_positions[block.cbz_layer]
                marker_y = self._bit_ypos(block.cbz_bit)
                dot = patches.Circle(
                    (marker_x, marker_y),
                    radius=0.18,
                    facecolor="k",
                    edgecolor="k",
                    zorder=PORDER_GATE,
                )
                self._ax.add_patch(dot)
                self._double_line(
                    (marker_x, marker_y),
                    (marker_x, y1),
                    lw=1.8,
                )
                self._text(
                    marker_x,
                    marker_y + 0.35,
                    block.label,
                    fontsize=20,
                    horizontalalignment="center",
                    verticalalignment="top",
                )

    def _line(
        self,
        from_xy: Tuple[float, float],
        to_xy: Tuple[float, float],
        lc: str = "k",
        ls: str = "-",
        lw: float = 2.0,
        zorder: int = PORDER_LINE,
    ) -> None:
        if self._ax is None:
            return
        from_x, from_y = from_xy
        to_x, to_y = to_xy
        self._ax.plot(
            [from_x, to_x],
            [from_y, to_y],
            color=lc,
            linestyle=ls,
            linewidth=lw,
            zorder=zorder,
        )

    def _double_line(
        self,
        from_xy: Tuple[float, float],
        to_xy: Tuple[float, float],
        lc: str = "k",
        lw: float = 2.0,
        zorder: int = PORDER_LINE,
        separation: float = 0.08,
    ) -> None:
        if self._ax is None:
            return
        from_x, from_y = from_xy
        to_x, to_y = to_xy
        dx = to_x - from_x
        dy = to_y - from_y
        length = hypot(dx, dy)
        if length == 0:
            self._line(from_xy, to_xy, lc=lc, lw=lw, zorder=zorder)
            return

        ox = -dy / length * separation
        oy = dx / length * separation
        offset1_from = (from_x + ox, from_y + oy)
        offset1_to = (to_x + ox, to_y + oy)
        offset2_from = (from_x - ox, from_y - oy)
        offset2_to = (to_x - ox, to_y - oy)

        self._line(offset1_from, offset1_to, lc=lc, lw=lw, zorder=zorder)
        self._line(offset2_from, offset2_to, lc=lc, lw=lw, zorder=zorder)

    def _text(
        self,
        x: float,
        y: float,
        text: str,
        horizontalalignment: str = "center",
        verticalalignment: str = "center",
        fontsize: int = 20,
        color: str = "k",
        clip_on: bool = True,
        zorder: int = PORDER_TEXT,
    ) -> None:
        if self._ax is None:
            return
        self._ax.text(
            x,
            y,
            text,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            fontsize=fontsize * self._fig_scale_factor,
            color=color,
            clip_on=clip_on,
            zorder=zorder,
        )

    def _gate_with_size(
        self, gate: GateData, xy: Tuple[float, float], multi_gate_size: int
    ) -> None:
        if self._ax is None:
            return
        xpos, ypos = xy
        gate_width = _calc_gate_width(gate)

        ypos = (
            ypos
            + (
                ypos
                + (multi_gate_size - 1)
                * (GATE_DEFAULT_HEIGHT + GATE_MARGIN_BOTTOM + GATE_MARGIN_TOP)
            )
        ) * 0.5
        multi_gate_height = GATE_DEFAULT_HEIGHT * multi_gate_size + (
            GATE_MARGIN_BOTTOM + GATE_MARGIN_TOP
        ) * (multi_gate_size - 1)
        facecolor = "w" if gate.kind is GateType.GENERIC else "#f7f7f7"
        box = patches.Rectangle(
            xy=(xpos - 0.5 * gate_width, ypos - 0.5 * multi_gate_height),
            width=gate_width,
            height=multi_gate_height,
            facecolor=facecolor,
            edgecolor="k",
            linewidth=2.4,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(box)

        if gate.name == "":
            latex_style_gate_str = ""
        else:
            try:
                latex_style_gate_str = f"${to_latex_style(gate.name)}$"
            except KeyError:
                latex_style_gate_str = gate.name

        self._text(xpos, ypos, latex_style_gate_str)
        self._control_bits(gate.control_bit_infos, (xpos, ypos))

    def _multi_gate(self, gate: GateData, xy: Tuple[float, float]) -> None:
        xpos, _ = xy
        multi_gate_data = GateData(gate.name, kind=gate.kind)
        groups_of_adjacent_gates = grouping_adjacent_gates(list(gate.all_target_bits))

        for i, adjacent_gates in enumerate(groups_of_adjacent_gates):
            group_x = xpos
            group_y = adjacent_gates[0] * self._wire_pitch
            self._gate_with_size(
                multi_gate_data, (group_x, group_y), len(adjacent_gates)
            )
            if i == 0:
                multi_gate_data.name = ""

        ypos = min(gate.all_target_bits) * self._wire_pitch
        to_ypos = max(gate.all_target_bits) * self._wire_pitch
        self._double_line((xpos, ypos), (xpos, to_ypos), lw=10, lc="gray")
        self._control_bits(gate.control_bit_infos, (xpos, ypos))

    def _control_bits(
        self, control_bit_infos: List[ControlBitInfo], xy_from: Tuple[float, float]
    ) -> None:
        if self._ax is None:
            return

        for info in control_bit_infos:
            control_bit = info.index
            if control_bit < self._circuit.qubit_count:
                continue
            to_ypos = self._bit_ypos(control_bit)
            to_xpos = xy_from[0]
            self._double_line(
                xy_from,
                (to_xpos, to_ypos),
                lw=1.8,
            )
            if info.control_value == 0:
                ctl = patches.Circle(
                    xy=(to_xpos, to_ypos),
                    radius=0.15,
                    fc="w",
                    ec="k",
                    linewidth=2.0,
                    zorder=PORDER_GATE,
                )
            else:
                ctl = patches.Circle(
                    xy=(to_xpos, to_ypos),
                    radius=0.15,
                    fc="k",
                    ec="w",
                    linewidth=0,
                    zorder=PORDER_GATE,
                )
            self._ax.add_patch(ctl)
