# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional

from ..evaluate import EvaluatorHooks
from ..machineinst import MachineSub, Primitive, SubId, is_subcall
from ..namespace import NameSpace
from ..op import AbstractOp, BaseIdent, Op
from ..qubit import Qubit
from ..register import Register


@dataclass(frozen=True)
class QubitCounts:
    """Qubits a sub occupies during one invocation.

    ``auxiliary`` is the peak ancilla count, i.e. the maximum over the
    sub's callees plus the ancillas the body allocates itself.
    """

    data: int
    auxiliary: int
    total: int


@dataclass(frozen=True)
class GateCounts:
    """Primitive-gate counts of one scope.

    Keys are bare local gate names, qualified as ``"ns.name"`` when two
    counted gates share a local name. ``per_call`` is for a single
    invocation of the sub, ``all_calls`` for all ``call_count`` of them.
    """

    per_call: Mapping[str, int]
    all_calls: Mapping[str, int]


@dataclass(frozen=True)
class GateCountBreakdown:
    """Primitive-gate counts of a sub, split by where the gates come from.

    ``own`` covers gates written directly in the body, ``descendants``
    those contributed transitively by everything the sub calls, and
    ``total`` their sum.
    """

    own: GateCounts
    descendants: GateCounts
    total: GateCounts


@dataclass(frozen=True)
class SubwiseProfileEntry:
    """Per-sub profiling row produced by :class:`SubwiseProfileEvaluatorHooks`.

    ``name`` is the readable identifier of the sub (e.g. ``"MCX(6)"``),
    or ``None`` when the sub's source op is unknown (the entry sub of
    ``compile_sub``, which receives a ``Sub`` rather than an ``Op``).
    """

    name: Optional[str]
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
    own_gates: Counter[BaseIdent] = field(default_factory=Counter)
    descendant_gates: Counter[BaseIdent] = field(default_factory=Counter)
    # Invocations of each descendant, the sub itself included.
    subtree_calls: Counter[SubId] = field(default_factory=Counter)
    peak_aux: int = 0

    def __post_init__(self) -> None:
        self.subtree_calls[self.sub.sub_id] = 1

    def record_gate(self, base_id: BaseIdent) -> None:
        self.own_gates[base_id] += 1

    def total_gates(self) -> Counter[BaseIdent]:
        return self.own_gates + self.descendant_gates

    def absorb_call(self, callee: "_SubProfile") -> None:
        """Fold one finished ``callee`` invocation into this subtree."""
        self.descendant_gates.update(callee.total_gates())
        self.subtree_calls.update(callee.subtree_calls)
        # Sequential callees reuse ancillas, so the peak is the max, not the sum.
        self.peak_aux = max(self.peak_aux, callee.peak_aux)

    def allocate_aux_qubits(self, count: int) -> None:
        """Add ancillas the body allocates itself.

        They stay live while the callees run, so they stack on top of
        the peak already absorbed from them.
        """
        self.peak_aux += count


class SubwiseProfileEvaluatorHooks(EvaluatorHooks[list[SubwiseProfileEntry]]):
    """Report, per distinct sub, how often it is called and the gate/qubit
    counts attributable to it.

    Mirrors :class:`GateCountEvaluatorHooks`: pass the ops to break
    down, or nothing to count every primitive. Parametric subs are
    reported separately per parameter (``MCX(5)`` and ``MCX(6)`` are
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
        if self._target_ops and mop.op.base_id not in self._target_ops:
            return
        self._profiles[call_stack[-1]].record_gate(mop.op.base_id)

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
        sub_ops = _source_ops(self._profiles.values())
        # One shared mapping so the same gate is keyed identically in every
        # row. Own gates suffice: every counted gate is own to some sub.
        name_map = _display_name_map(
            base_id
            for profile in self._profiles.values()
            for base_id in profile.own_gates
        )
        entries = []
        for sid, profile in self._profiles.items():
            call_count = root.subtree_calls.get(sid, 0)
            entries.append(
                SubwiseProfileEntry(
                    name=_format_name(sub_ops.get(sid)),
                    call_count=call_count,
                    qubit_counts=_qubit_counts(profile),
                    gate_counts=_gate_counts(profile, name_map, call_count),
                )
            )
        return entries


def _source_ops(profiles: Iterable[_SubProfile]) -> dict[SubId, Op]:
    """Collect each sub's source Op for naming.

    A sub's Op is known at its call sites rather than on the sub itself
    (only the compile entry carries its own), so harvest the subcall
    instructions of every profiled sub.
    """
    ops: dict[SubId, Op] = {}
    for profile in profiles:
        if profile.sub.op is not None:
            ops[profile.sub.sub_id] = profile.sub.op
        for mop, _, _ in profile.sub.instructions:
            if is_subcall(mop) and mop.sub is not None:
                ops[mop.sub.sub_id] = mop.op
    return ops


def _qubit_counts(profile: _SubProfile) -> QubitCounts:
    data = len(profile.sub.qubits)
    return QubitCounts(
        data=data, auxiliary=profile.peak_aux, total=data + profile.peak_aux
    )


def _gate_counts(
    profile: _SubProfile, name_map: Mapping[BaseIdent, str], call_count: int
) -> GateCountBreakdown:
    own = _display_names(profile.own_gates, name_map)
    descendants = _display_names(profile.descendant_gates, name_map)
    total = _display_names(profile.total_gates(), name_map)
    return GateCountBreakdown(
        own=_aggregate(own, call_count),
        descendants=_aggregate(descendants, call_count),
        total=_aggregate(total, call_count),
    )


def _aggregate(per_call: Mapping[str, int], call_count: int) -> GateCounts:
    return GateCounts(
        per_call=dict(per_call),
        all_calls={name: n * call_count for name, n in per_call.items()},
    )


def _display_name_map(base_ids: Iterable[BaseIdent]) -> dict[BaseIdent, str]:
    """Map each gate to its bare local name, namespace-qualified only when two
    gates share a local name (bare names would silently merge their counts)."""
    namespaces_by_local: dict[str, set[NameSpace]] = defaultdict(set)
    ids = set(base_ids)
    for ns, local in ids:
        namespaces_by_local[local].add(ns)
    return {
        (ns, local): local if len(namespaces_by_local[local]) == 1 else f"{ns}.{local}"
        for ns, local in ids
    }


def _display_names(
    counts: Mapping[BaseIdent, int], name_map: Mapping[BaseIdent, str]
) -> dict[str, int]:
    return {name_map[base_id]: n for base_id, n in counts.items()}


def _format_name(op: Optional[Op]) -> Optional[str]:
    if op is None:
        return None
    if op.id.params:
        params = ", ".join(str(p) for p in op.id.params)
        return f"{op.id.local_name}({params})"
    return op.id.local_name
