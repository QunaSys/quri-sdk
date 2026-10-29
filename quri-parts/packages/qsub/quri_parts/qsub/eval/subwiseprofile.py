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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional

from ..evaluate import EvaluatorHooks
from ..machineinst import MachineSub, Primitive, SubId
from ..op import AbstractOp, Ident
from ..qubit import Qubit
from ..register import Register


@dataclass(frozen=True)
class QubitCounts:
    """Qubits a sub occupies during one invocation.

    ``data`` are the qubits the caller passes in. ``auxiliary`` are the
    sub's own ancillas (its ``aux_qubits``) plus the largest
    ``auxiliary`` among the subs it calls: sequential callees reuse the
    same ancilla space, so they contribute a maximum rather than a sum.
    The sub's own ancillas stay live across those calls, so they add to
    that maximum.
    """

    data: int
    auxiliary: int
    total: int


@dataclass(frozen=True)
class GateCounts:
    """Primitive-gate counts of one scope.

    Keys are gate ``Ident``\\ s, op parameters included, so ``RZ<0.1>``
    and ``RZ<0.2>`` are counted apart; this is finer than
    :class:`GateCountEvaluatorHooks`, which keys by ``BaseIdent`` and
    merges them. ``per_call`` is for a single invocation of the sub,
    ``all_calls`` for all ``call_count`` of them.
    """

    per_call: Mapping[Ident, int]
    all_calls: Mapping[Ident, int]


@dataclass(frozen=True)
class GateCountBreakdown:
    """Primitive-gate counts of a sub, split by where the gates come from.

    ``own`` covers the sub's own instructions, ``descendants`` those
    contributed transitively by everything it calls, and ``cumulative``
    their sum.
    """

    own: GateCounts
    descendants: GateCounts
    cumulative: GateCounts


@dataclass(frozen=True)
class SubwiseProfileEntry:
    """Per-sub profiling row produced by :class:`SubwiseProfileEvaluatorHooks`.

    ``source_op`` identifies the op this sub implements, parameters
    included; use :meth:`Ident.to_str` for a display name. It is
    ``None`` when the op is unknown (the entry sub of ``compile_sub``,
    which receives a ``Sub`` rather than an ``Op``).
    """

    source_op: Optional[Ident]
    call_count: int
    qubit_counts: QubitCounts
    gate_counts: GateCountBreakdown


@dataclass
class _SubProfile:
    """Counts accumulated for one distinct sub while its body is walked.

    Everything here is per single invocation: a sub's body is walked
    once, and each later invocation folds the accumulated numbers into
    its parent again.
    """

    sub: MachineSub
    own_gates: Counter[Ident] = field(default_factory=Counter)
    descendant_gates: Counter[Ident] = field(default_factory=Counter)
    # Invocations of each descendant, the sub itself included.
    subtree_calls: Counter[SubId] = field(default_factory=Counter)
    peak_aux: int = 0

    def __post_init__(self) -> None:
        self.subtree_calls[self.sub.sub_id] = 1

    def record_gate(self, ident: Ident) -> None:
        self.own_gates[ident] += 1

    def cumulative_gates(self) -> Counter[Ident]:
        return self.own_gates + self.descendant_gates

    def absorb_call(self, callee: "_SubProfile") -> None:
        """Fold one finished ``callee`` invocation into this subtree."""
        self.descendant_gates.update(callee.cumulative_gates())
        self.subtree_calls.update(callee.subtree_calls)
        # Sequential callees reuse ancillas, so the peak is the max, not the sum.
        self.peak_aux = max(self.peak_aux, callee.peak_aux)

    def allocate_aux_qubits(self, count: int) -> None:
        """Add the ancillas this sub declares itself.

        They stay live while the callees run, so they stack on top of
        the peak already absorbed from them.
        """
        self.peak_aux += count


class SubwiseProfileEvaluatorHooks(EvaluatorHooks[list[SubwiseProfileEntry]]):
    """Report, per distinct sub, how often it is called and the gate/qubit
    counts attributable to it.

    Takes the same arguments as :class:`GateCountEvaluatorHooks`: pass
    the ops to break down, or nothing to count every primitive. A row's
    ``source_op`` carries the op parameters, so parametric subs are
    reported separately per parameter (``MCX<5>`` and ``MCX<6>`` are
    distinct rows).
    """

    def __init__(self, gates: Iterable[AbstractOp] = ()) -> None:
        self._target_ops = {op.base_id for op in gates}
        self.reset()

    def reset(self) -> None:
        # Insertion order is the order of the reported rows.
        self._profiles: dict[SubId, _SubProfile] = {}
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
        if sub.sub_id in self._profiles:
            return False
        self._profiles[sub.sub_id] = _SubProfile(sub)
        return True

    def primitive(
        self,
        mop: Primitive,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> None:
        # Selection is by gate kind, but the count is kept per parameter.
        if self._target_ops and mop.op.base_id not in self._target_ops:
            return
        self._profiles[call_stack[-1]].record_gate(mop.op.id)

    def exit_sub(
        self, sub: MachineSub, enter_sub: bool, call_stack: list[SubId]
    ) -> None:
        profile = self._profiles[sub.sub_id]
        if enter_sub:
            profile.allocate_aux_qubits(len(sub.aux_qubits))

        if len(call_stack) > 1:
            self._profiles[call_stack[-2]].absorb_call(profile)
        else:
            self._result = self._build_entries(root=profile)

    def _build_entries(self, root: _SubProfile) -> list[SubwiseProfileEntry]:
        entries = []
        for sid, profile in self._profiles.items():
            call_count = root.subtree_calls.get(sid, 0)
            entries.append(
                SubwiseProfileEntry(
                    source_op=profile.sub.source_op_id,
                    call_count=call_count,
                    qubit_counts=_qubit_counts(profile),
                    gate_counts=_gate_counts(profile, call_count),
                )
            )
        return entries


def _qubit_counts(profile: _SubProfile) -> QubitCounts:
    data = len(profile.sub.qubits)
    return QubitCounts(
        data=data, auxiliary=profile.peak_aux, total=data + profile.peak_aux
    )


def _gate_counts(profile: _SubProfile, call_count: int) -> GateCountBreakdown:
    return GateCountBreakdown(
        own=_aggregate(profile.own_gates, call_count),
        descendants=_aggregate(profile.descendant_gates, call_count),
        cumulative=_aggregate(profile.cumulative_gates(), call_count),
    )


def _aggregate(per_call: Mapping[Ident, int], call_count: int) -> GateCounts:
    return GateCounts(
        per_call=dict(per_call),
        all_calls={ident: n * call_count for ident, n in per_call.items()},
    )
