# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Mapping, Optional, TypeAlias

from quri_parts.qsub._visualization import (
    CircuitData,
    ConditionalBlock,
    ControlBitInfo,
    GateData,
    GateType,
    MPLCircuitlDrawer,
)
from quri_parts.qsub.eval.subwiseprofile import SubwiseProfileEntry
from quri_parts.qsub.lib.std import (
    CNOT,
    CZ,
    Cbz,
    Controlled,
    Label,
    M,
    MultiControlled,
    Toffoli,
)
from quri_parts.qsub.machineinst import MachineSub
from quri_parts.qsub.op import Ident, Op
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.register import Register
from quri_parts.qsub.resolve import SubRepository, default_repository, resolve_sub
from quri_parts.qsub.sub import Sub, SubBuilder

if TYPE_CHECKING:
    import graphviz
    import matplotlib


__all__ = [
    "CircuitData",
    "ConditionalBlock",
    "ControlBitInfo",
    "GateData",
    "GateType",
    "MPLCircuitlDrawer",
    "draw",
    "draw_call_graph",
    "draw_msub",
    "draw_sub",
    "machine_sub_to_vis_data",
    "op_to_vis_data",
    "sub_to_vis_data",
]


#: A control qubit specified by (qubit_index, control_value).
#: control_value should be 0 or 1.
ControlQubit: TypeAlias = tuple[int, int]


def _op_controls(op: Op, offset: int = 0) -> Collection[ControlQubit]:
    match op.base_id:
        case CNOT.base_id | CZ.base_id | Controlled.base_id:
            return ((offset, 1),)
        case Toffoli.base_id:
            return ((offset, 1), (offset + 1, 1))
        case MultiControlled.base_id:
            _, control_bits, control_value = op.id.params
            assert isinstance(control_bits, int)
            assert isinstance(control_value, int)
            return tuple(
                (offset + i, (control_value >> i) & 1) for i in range(control_bits)
            )
    return ()


def op_to_vis_data(
    op: Op,
    qubits: Sequence[Qubit],
    regs: Sequence[Register],
    register_indices: Optional[Mapping[int, int]] = None,
) -> GateData:
    """Convert an Op to data used by the visualization layer."""
    if register_indices is None:
        register_indices = {}
    offset = 0
    controls = _op_controls(op, offset)
    while op.base_id == Controlled.base_id or op.base_id == MultiControlled.base_id:
        _op = op.id.params[0]
        assert isinstance(_op, Op)
        op = _op
        offset = len(controls)
        controls = (*controls, *_op_controls(op, offset))

    name = op.id.to_str(full=False, param_truncate=8)
    control_info = [ControlBitInfo(qubits[c[0]].uid, c[1]) for c in controls]
    control_indices = [c[0] for c in controls]
    target_bits = [
        qubits[i].uid for i in range(len(qubits)) if i not in control_indices
    ]
    classical_targets: list[int] = []
    gate_type = GateType.GENERIC

    if op.base_id == M.base_id and regs:
        if regs[0].uid in register_indices:
            classical_targets.append(register_indices[regs[0].uid])
        gate_type = GateType.MEASURE
    elif op.base_id == Cbz.base_id and len(regs) >= 2:
        ctrl_idx = register_indices.get(regs[0].uid)
        if ctrl_idx is not None:
            classical_targets.append(ctrl_idx)
        target_bits = []
        gate_type = GateType.CLASSICAL
    elif op.base_id == Label.base_id and regs:
        if regs[0].uid in register_indices:
            classical_targets.append(register_indices[regs[0].uid])
        target_bits = []
        gate_type = GateType.CLASSICAL
    else:
        classical_targets.extend(register_indices.get(reg.uid, -1) for reg in regs)
        classical_targets = [c for c in classical_targets if c >= 0]
        if not target_bits and classical_targets:
            gate_type = GateType.CLASSICAL

    return GateData(
        name,
        target_bits,
        control_info,
        classical_targets,
        gate_type,
    )


def sub_to_vis_data(sub: Sub) -> CircuitData:
    """Convert a Sub to data used by the visualization layer."""
    qubit_count = len(sub.qubits) + len(sub.aux_qubits)
    all_registers = (*sub.registers, *sub.aux_registers)
    registers_to_draw: list[Register] = []
    used_reg_uids: set[int] = set()
    for op, _qubits, regs in sub.operations:
        if op.base_id == Label.base_id:
            continue
        if op.base_id == Cbz.base_id and len(regs) >= 2:
            used_reg_uids.add(regs[0].uid)
            continue
        used_reg_uids.update(r.uid for r in regs)
    for reg in all_registers:
        if reg.uid in used_reg_uids:
            registers_to_draw.append(reg)

    register_indices = {
        reg.uid: qubit_count + i for i, reg in enumerate(registers_to_draw)
    }
    gates: list[GateData] = []
    condition_pairs: list[dict[str, object]] = []

    open_conditions: list[dict[str, object]] = []
    for op, qubits, regs in sub.operations:
        if not qubits and not regs:
            continue
        gate = op_to_vis_data(op, qubits, regs, register_indices)
        gates.append(gate)

        if op.base_id == Cbz.base_id and len(regs) >= 2:
            cond_pair = {
                "start_gate": gate,
                "cond_reg": regs[0],
                "label_reg": regs[1],
                "body_first_gate": None,
                "body_last_gate": None,
            }
            condition_pairs.append(cond_pair)
            open_conditions.append(cond_pair)
        if op.base_id == Label.base_id and regs:
            for pair in condition_pairs:
                if pair.get("label_reg") == regs[0] and "end_gate" not in pair:
                    end_gate = pair.get("body_last_gate") or gate
                    pair["end_gate"] = end_gate
                    break
        for pair in open_conditions:
            if (
                pair.get("end_gate") is None
                and gate is not pair.get("start_gate")
                and op.base_id != Label.base_id
            ):
                if pair.get("body_first_gate") is None:
                    pair["body_first_gate"] = gate
                pair["body_last_gate"] = gate

    circuit = CircuitData.from_gate_sequence(
        gates,
        qubit_count=qubit_count,
        register_count=len(registers_to_draw),
    )
    aux_reg_uids = {r.uid for r in sub.aux_registers}
    circuit.conditional_blocks = _build_conditional_blocks(
        circuit,
        condition_pairs,
        register_indices,
        excluded_bits=[
            idx for reg_uid, idx in register_indices.items() if reg_uid in aux_reg_uids
        ],
    )
    return circuit


def machine_sub_to_vis_data(msub: MachineSub) -> CircuitData:
    """Convert a MachineSub to data used by the visualization layer."""
    qubit_count = len(msub.qubits) + len(msub.aux_qubits)
    all_registers = (*msub.registers, *msub.aux_registers)
    registers_to_draw: list[Register] = []
    used_reg_uids: set[int] = set()
    for mop, _qubits, regs in msub.instructions:
        if mop.op.base_id == Label.base_id:
            continue
        if mop.op.base_id == Cbz.base_id and len(regs) >= 2:
            used_reg_uids.add(regs[0].uid)
            continue
        used_reg_uids.update(r.uid for r in regs)
    for reg in all_registers:
        if reg.uid in used_reg_uids:
            registers_to_draw.append(reg)

    register_indices = {
        reg.uid: qubit_count + i for i, reg in enumerate(registers_to_draw)
    }
    gates: list[GateData] = []
    condition_pairs: list[dict[str, object]] = []
    for mop, qubits, regs in msub.instructions:
        if not qubits and not regs:
            continue
        gate = op_to_vis_data(mop.op, qubits, regs, register_indices)
        gates.append(gate)

        if mop.op.base_id == Cbz.base_id and len(regs) >= 2:
            condition_pairs.append(
                {
                    "start_gate": gate,
                    "cond_reg": regs[0],
                    "label_reg": regs[1],
                    "body_first_gate": None,
                    "body_last_gate": None,
                }
            )
        if mop.op.base_id == Label.base_id and regs:
            for pair in condition_pairs:
                if pair.get("label_reg") == regs[0] and "end_gate" not in pair:
                    pair["end_gate"] = gate
                    break
        for pair in condition_pairs:
            if (
                pair.get("end_gate") is None
                and gate is not pair.get("start_gate")
                and mop.op.base_id != Label.base_id
            ):
                if pair.get("body_first_gate") is None:
                    pair["body_first_gate"] = gate
                pair["body_last_gate"] = gate

    circuit = CircuitData.from_gate_sequence(
        gates,
        qubit_count=qubit_count,
        register_count=len(registers_to_draw),
    )
    aux_reg_uids = {r.uid for r in msub.aux_registers}
    circuit.conditional_blocks = _build_conditional_blocks(
        circuit,
        condition_pairs,
        register_indices,
        excluded_bits=[
            idx for reg_uid, idx in register_indices.items() if reg_uid in aux_reg_uids
        ],
    )
    return circuit


def _find_gate_layer(circuit: CircuitData, gate: GateData) -> int:
    for layer in range(circuit.layer_count):
        for bit in range(circuit.bit_count):
            if circuit.gates[bit][layer] is gate:
                return layer
    raise ValueError("Gate not found in circuit data")


def _build_conditional_blocks(
    circuit: CircuitData,
    condition_pairs: Sequence[Mapping[str, object]],
    register_indices: Mapping[int, int],
    excluded_bits: Sequence[int] = (),
) -> list[ConditionalBlock]:
    blocks: list[ConditionalBlock] = []
    excluded_set = set(excluded_bits)
    for pair in condition_pairs:
        start_gate = pair.get("start_gate")
        end_gate = pair.get("end_gate")
        body_first_gate = pair.get("body_first_gate")
        body_last_gate = pair.get("body_last_gate")
        cond_reg = pair.get("cond_reg")
        label_reg = pair.get("label_reg")
        if not isinstance(start_gate, GateData):
            continue
        assert isinstance(cond_reg, Register)
        assert isinstance(label_reg, Register)
        try:
            cbz_layer = _find_gate_layer(circuit, start_gate)
        except ValueError:
            continue
        start_layer: int | None
        end_layer: int | None
        if isinstance(body_first_gate, GateData):
            try:
                start_layer = _find_gate_layer(circuit, body_first_gate)
            except ValueError:
                start_layer = None
        else:
            start_layer = None
        if isinstance(body_last_gate, GateData):
            try:
                end_layer = _find_gate_layer(circuit, body_last_gate)
            except ValueError:
                end_layer = start_layer
        elif isinstance(end_gate, GateData):
            try:
                end_layer = _find_gate_layer(circuit, end_gate)
            except ValueError:
                end_layer = start_layer
        else:
            end_layer = start_layer

        if start_layer is None or end_layer is None:
            continue
        if start_layer > end_layer:
            start_layer, end_layer = end_layer, start_layer

        bits: set[int] = set()
        cond_idx = register_indices.get(cond_reg.uid)
        if cond_idx is not None and cond_idx not in excluded_set:
            bits.add(cond_idx)
        label_idx = register_indices.get(label_reg.uid)
        if label_idx is not None and label_idx not in excluded_set:
            bits.add(label_idx)
        for layer in range(start_layer, end_layer + 1):
            for bit_index, line in enumerate(circuit.gates):
                if layer >= len(line):
                    continue
                gate = line[layer]
                if gate.name in {"wire", "ghost"}:
                    continue
                bits.update(b for b in gate.all_target_bits if b not in excluded_set)
                bits.update(
                    c.index
                    for c in gate.control_bit_infos
                    if c.index not in excluded_set
                )

        if cond_idx is not None:
            visible_cond = cond_idx - circuit.qubit_count
        else:
            visible_cond = None
        label = f"r{visible_cond} == 0" if visible_cond is not None else "cond == 0"
        if not bits:
            continue
        blocks.append(
            ConditionalBlock(
                start_layer=start_layer,
                end_layer=end_layer,
                bits=sorted(bits),
                label=label,
                cbz_layer=cbz_layer,
                cbz_bit=cond_idx,
            )
        )

    return blocks


def draw_sub(
    sub: Op | Sub,
    *,
    dpi: int = 72,
    scale: float = 0.6,
    debug: bool = False,
    filename: Optional[str] = None,
    repository: SubRepository = default_repository(),
) -> "matplotlib.figure.Figure":
    """Draw a diagram for a given Sub."""
    if isinstance(sub, Op):
        s = resolve_sub(sub, repository)
        if s is None:
            builder = SubBuilder(sub.qubit_count, sub.reg_count)
            builder.add_op(sub, builder.qubits, builder.registers)
            s = builder.build()
    else:
        s = sub
    return MPLCircuitlDrawer(sub_to_vis_data(s), dpi=dpi, scale=scale).draw(
        debug=debug, filename=filename
    )


def draw_msub(
    msub: MachineSub,
    *,
    dpi: int = 72,
    scale: float = 0.6,
    debug: bool = False,
    filename: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Draw a diagram for a given MachineSub."""
    return MPLCircuitlDrawer(machine_sub_to_vis_data(msub), dpi=dpi, scale=scale).draw(
        debug=debug, filename=filename
    )


def draw(
    sub: Op | Sub | MachineSub,
    *,
    dpi: int = 72,
    scale: float = 0.6,
    debug: bool = False,
    filename: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Draw a diagram for a given Sub or MachineSub."""
    if isinstance(sub, (Op, Sub)):
        return draw_sub(sub, dpi=dpi, scale=scale, debug=debug, filename=filename)
    else:
        return draw_msub(sub, dpi=dpi, scale=scale, debug=debug, filename=filename)


def _call_graph_node_names(
    nodes: Iterable[Optional[Ident]],
) -> dict[Optional[Ident], str]:
    """Name each node by its short ident, falling back to the fully qualified
    one where truncation would make distinct subs look alike."""
    short = {
        node: node.to_str(full=False, param_truncate=8)
        if node is not None
        else "(unnamed)"
        for node in nodes
    }
    collisions = {name for name, n in Counter(short.values()).items() if n > 1}
    return {
        node: node.to_str(full=True)
        if node is not None and name in collisions
        else name
        for node, name in short.items()
    }


def _call_graph_node_label(name: str, entry: Optional[SubwiseProfileEntry]) -> str:
    if entry is None:
        return name
    lines = [name, f"calls: {entry.call_count}"]
    lines.extend(
        f"{gate.to_str(full=False, param_truncate=8)}: {count}"
        for gate, count in entry.gate_counts.cumulative.all_calls.items()
    )
    # DOT's escape for a centered line break.
    return "\\n".join(lines)


def draw_call_graph(
    graph: Mapping[Optional[Ident], Mapping[Ident, int]],
    profile: Optional[Sequence[SubwiseProfileEntry]] = None,
    *,
    filename: Optional[str] = None,
) -> "graphviz.Digraph":
    """Draw a call graph produced by :class:`CallGraphEvaluatorHooks`, laid out
    top-down by Graphviz.

    Each edge is annotated with its number of calls per caller
    invocation, highlighted when greater than one. Pass the report of
    :class:`SubwiseProfileEvaluatorHooks` for the same compiled sub as
    ``profile`` to also annotate each node with its call count and
    cumulative gate counts (the two reports join on the op ``Ident``:
    graph nodes against profile ``source_op``).

    The returned :class:`graphviz.Digraph` renders inline in Jupyter.
    Pass ``filename`` to also save the drawing. The file extension sets
    the format, for example ``.svg``, ``.png`` or ``.pdf``.

    The ``graphviz`` extra (``pip install "quri-parts-qsub[graphviz]"``)
    provides the Python package. Rendering also needs the Graphviz
    system package (https://graphviz.org/download/).
    """
    try:
        import graphviz
    except ImportError as e:
        raise ImportError(
            "draw_call_graph requires the graphviz package: "
            "pip install 'quri-parts-qsub[graphviz]'. Rendering also needs the "
            "Graphviz system package: https://graphviz.org/download/"
        ) from e

    entries: Mapping[Optional[Ident], SubwiseProfileEntry] = (
        {e.source_op: e for e in profile} if profile is not None else {}
    )
    nodes = list(
        dict.fromkeys(
            [*graph, *(callee for callees in graph.values() for callee in callees)]
        )
    )
    names = _call_graph_node_names(nodes)
    # graphviz reads ':' in an edge endpoint as a port separator, so nodes get
    # plain ids and show their ident strings only as labels.
    node_ids = {node: f"n{i}" for i, node in enumerate(nodes)}

    dot = graphviz.Digraph()
    dot.attr(
        "node",
        shape="box",
        style="rounded",
        # dot estimates text width from its own font metrics, which can run
        # narrower than the font that finally renders; pad so labels fit.
        margin="0.15,0.06",
        color="#808080",
        fontname="Helvetica,Arial,sans-serif",
        fontsize="11",
    )
    dot.attr(
        "edge",
        color="#808080",
        arrowsize="0.7",
        fontname="Helvetica,Arial,sans-serif",
        fontsize="10",
    )
    for node in nodes:
        dot.node(node_ids[node], _call_graph_node_label(names[node], entries.get(node)))
    for caller, callees in graph.items():
        for callee, m in callees.items():
            dot.edge(
                node_ids[caller],
                node_ids[callee],
                label=f"x{m}",
                fontcolor="#d62728" if m > 1 else "#808080",
            )
    if filename:
        dot.render(outfile=filename, cleanup=True)
    return dot
