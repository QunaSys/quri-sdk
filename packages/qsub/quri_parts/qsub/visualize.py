# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Optional, TypeAlias

from qulacsvis.models.circuit import (  # type: ignore
    CircuitData,
    ControlQubitInfo,
    GateData,
)
from qulacsvis.visualization import MPLCircuitlDrawer  # type: ignore

from quri_parts.qsub.lib.std import CNOT, CZ, Controlled, MultiControlled, Toffoli
from quri_parts.qsub.machineinst import MachineSub
from quri_parts.qsub.op import Op
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.register import Register
from quri_parts.qsub.resolve import SubRepository, default_repository, resolve_sub
from quri_parts.qsub.sub import Sub

if TYPE_CHECKING:
    import matplotlib


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
    op: Op, qubits: Sequence[Qubit], regs: Sequence[Register]
) -> GateData:
    """Convert an Op to data used by qulacsvis."""
    offset = 0
    controls = _op_controls(op, offset)
    while op.base_id == Controlled.base_id or op.base_id == MultiControlled.base_id:
        _op = op.id.params[0]
        assert isinstance(_op, Op)
        op = _op
        offset = len(controls)
        controls = (*controls, *_op_controls(op, offset))

    name = op.id.to_str(full=False, param_truncate=8)
    control_info = [ControlQubitInfo(qubits[c[0]].uid, c[1]) for c in controls]
    control_indices = [c[0] for c in controls]
    target_bits = [
        qubits[i].uid for i in range(len(qubits)) if i not in control_indices
    ]
    return GateData(name, target_bits, control_info)


def sub_to_vis_data(sub: Sub) -> CircuitData:
    """Convert a Sub to data used by qulacsvis."""
    gates = [op_to_vis_data(op, qubits, regs) for op, qubits, regs in sub.operations]
    return CircuitData.from_gate_sequence(gates, len(sub.qubits) + len(sub.aux_qubits))


def machine_sub_to_vis_data(msub: MachineSub) -> CircuitData:
    """Convert a MachineSub to data used by qulacsvis."""
    gates = [
        op_to_vis_data(mop.op, qubits, regs) for mop, qubits, regs in msub.instructions
    ]
    return CircuitData.from_gate_sequence(
        gates, len(msub.qubits) + len(msub.aux_qubits)
    )


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
            raise KeyError(f"Sub not found for Op: {sub}")
    else:
        s = sub
    return MPLCircuitlDrawer(  # type: ignore
        sub_to_vis_data(s), dpi=dpi, scale=scale
    ).draw(debug=debug, filename=filename)


def draw_msub(
    msub: MachineSub,
    *,
    dpi: int = 72,
    scale: float = 0.6,
    debug: bool = False,
    filename: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Draw a diagram for a given MachineSub."""
    return MPLCircuitlDrawer(  # type: ignore
        machine_sub_to_vis_data(msub), dpi=dpi, scale=scale
    ).draw(debug=debug, filename=filename)


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
        return draw_sub(sub)
    else:
        return draw_msub(sub)
