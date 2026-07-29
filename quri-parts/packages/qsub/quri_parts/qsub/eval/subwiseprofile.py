# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from ..evaluate import EvaluatorHooks
from ..machineinst import MachineSub, Primitive, SubId, is_subcall
from ..op import AbstractOp, BaseIdent, Op
from ..qubit import Qubit
from ..register import Register


@dataclass(frozen=True)
class SubwiseProfileEntry:
    """Per-sub profiling row produced by :class:`SubwiseProfileEvaluatorHooks`.

    ``name`` is the readable identifier of the sub (e.g. ``"MCX(6)"``), or
    ``None`` when the sub's source op is unknown (the entry sub of
    ``compile_sub``, which receives a ``Sub`` rather than an ``Op``).

    ``qubit_count_info`` holds ``data`` (input/output width), ``auxiliary``
    (peak ancilla required for one invocation), and their sum ``total``.

    ``gate_count_info`` breaks the primitive-gate counts down by scope
    (``self`` for gates written directly in the body, ``descendants`` for gates
    contributed transitively by everything the sub calls, ``total`` for their
    sum) crossed with aggregation: the ``*_per_call`` keys are per single
    invocation, the unsuffixed keys are multiplied by ``call_count``.
    """

    name: Optional[str]
    call_count: int
    qubit_count_info: dict[str, int]
    gate_count_info: dict[str, dict[str, int]]


class SubwiseProfileEvaluatorHooks(EvaluatorHooks[list[SubwiseProfileEntry]]):
    """Report, per distinct sub, how often it is called and the gate/qubit
    counts attributable to it.

    Mirrors :class:`GateCountEvaluatorHooks`: pass the ops to break
    down, or nothing to count every primitive. Parametric subs are
    reported separately per parameter (``MCX(5)`` and ``MCX(6)`` are
    distinct rows).
    """

    def __init__(self, gates: Iterable[AbstractOp] = ()) -> None:
        self._target_ops = set([op.base_id for op in gates])
        self.reset()

    def reset(self) -> None:
        self._self_gates: dict[SubId, dict[BaseIdent, int]] = {}
        self._total_gates: dict[SubId, dict[BaseIdent, int]] = {}
        self._aux: dict[SubId, int] = defaultdict(int)
        self._call_mult: dict[SubId, dict[SubId, int]] = {}
        self._sub_op: dict[SubId, Op] = {}
        self._sub_by_id: dict[SubId, MachineSub] = {}
        self._order: list[SubId] = []
        self._entered: set[SubId] = set()
        self._result: list[SubwiseProfileEntry] = []

    def result(self) -> list[SubwiseProfileEntry]:
        return self._result

    def enter_sub(
        self,
        sub: MachineSub,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> bool:
        sid = sub.sub_id
        if sid in self._entered:
            return False

        self._entered.add(sid)
        self._order.append(sid)
        self._sub_by_id[sid] = sub
        self._self_gates[sid] = defaultdict(int)
        self._total_gates[sid] = defaultdict(int)
        self._call_mult[sid] = defaultdict(int)
        self._call_mult[sid][sid] = 1

        if sub.op is not None:
            self._sub_op[sid] = sub.op
        for mop, _, _ in sub.instructions:
            if is_subcall(mop) and mop.sub is not None:
                self._sub_op[mop.sub.sub_id] = mop.op

        return True

    def primitive(
        self,
        mop: Primitive,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> None:
        if self._target_ops and mop.op.base_id not in self._target_ops:
            return None
        sid = call_stack[-1]
        self._self_gates[sid][mop.op.base_id] += 1
        self._total_gates[sid][mop.op.base_id] += 1

    def exit_sub(
        self, sub: MachineSub, enter_sub: bool, call_stack: list[SubId]
    ) -> None:
        sid = sub.sub_id
        if enter_sub:
            self._aux[sid] += len(sub.aux_qubits)

        if len(call_stack) > 1:
            parent = call_stack[-2]
            for k, v in self._total_gates[sid].items():
                self._total_gates[parent][k] += v
            for cid, v in self._call_mult[sid].items():
                self._call_mult[parent][cid] += v
            self._aux[parent] = max(self._aux[parent], self._aux[sid])
        else:
            self._assemble(sid)

    def _assemble(self, root_sid: SubId) -> None:
        call_counts = self._call_mult[root_sid]
        entries: list[SubwiseProfileEntry] = []
        for sid in self._order:
            sub = self._sub_by_id[sid]
            cc = call_counts.get(sid, 0)

            data = len(sub.qubits)
            aux = self._aux[sid]
            qubit_count_info = {"data": data, "auxiliary": aux, "total": data + aux}

            self_pc = _display_names(self._self_gates[sid])
            total_pc = _display_names(self._total_gates[sid])
            desc_pc = {
                k: total_pc[k] - self_pc.get(k, 0)
                for k in total_pc
                if total_pc[k] - self_pc.get(k, 0) > 0
            }
            gate_count_info = {
                "self_per_call": self_pc,
                "self": _scale(self_pc, cc),
                "descendants_per_call": desc_pc,
                "descendants": _scale(desc_pc, cc),
                "total_per_call": total_pc,
                "total": _scale(total_pc, cc),
            }

            entries.append(
                SubwiseProfileEntry(
                    name=_format_name(self._sub_op.get(sid)),
                    call_count=cc,
                    qubit_count_info=qubit_count_info,
                    gate_count_info=gate_count_info,
                )
            )
        self._result = entries


def _display_names(counts: Mapping[BaseIdent, int]) -> dict[str, int]:
    """Render gate keys as their bare local name (dropping the namespace),
    matching the convention used elsewhere in the qsub counting reports."""
    return {base_id[1]: n for base_id, n in counts.items()}


def _scale(counts: Mapping[str, int], factor: int) -> dict[str, int]:
    return {k: v * factor for k, v in counts.items()}


def _format_name(op: Optional[Op]) -> Optional[str]:
    if op is None:
        return None
    if op.id.params:
        params = ", ".join(str(p) for p in op.id.params)
        return f"{op.id.local_name}({params})"
    return op.id.local_name
