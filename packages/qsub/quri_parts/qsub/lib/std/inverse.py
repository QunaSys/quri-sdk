# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection

from quri_parts.qsub.lib.std import Controlled, MultiControlled
from quri_parts.qsub.op import (
    AbstractOp,
    Ident,
    Op,
    OpFactory,
    ParameterValidationError,
    ParamUnitaryDef,
    param_op,
)
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.register import Register
from quri_parts.qsub.resolve import (
    SubRepository,
    SubResolver,
    SubResolverCondition,
    default_repository,
)
from quri_parts.qsub.sub import Sub, SubBuilder

from . import NS
from .rotation import RX, RY, RZ, Phase
from .single_clifford import S, Sdag, SqrtX, SqrtXdag, SqrtY, SqrtYdag
from .t import T, Tdag


class _Inverse(ParamUnitaryDef[Op]):
    ns = NS
    name = "Inverse"

    def qubit_count_fn(self, target_op: Op) -> int:
        return target_op.qubit_count

    def reg_count_fn(self, target_op: Op) -> int:
        return target_op.reg_count

    def validate_params(self, target_op: Op) -> None:
        if not target_op.unitary:
            raise ParameterValidationError(f"target_op {target_op} is not unitary")


Inverse = param_op(_Inverse)


def _single_op_sub(op: Op) -> Sub:
    b = SubBuilder(op.qubit_count, op.reg_count)
    b.add_op(op, b.qubits, b.registers)
    return b.build()


def _copy_sub_skeleton(
    target_sub: Sub,
) -> tuple[SubBuilder, dict[Qubit, Qubit], dict[Register, Register]]:
    qubit_count = len(target_sub.qubits)
    reg_count = len(target_sub.registers)
    builder = SubBuilder(qubit_count, reg_count)
    target_q = builder.qubits
    target_aq = tuple(builder.add_aux_qubit() for _ in target_sub.aux_qubits)
    qubit_map = dict(zip(target_sub.qubits, target_q)) | dict(
        zip(target_sub.aux_qubits, target_aq)
    )
    target_ar = tuple(builder.add_aux_register() for _ in target_sub.aux_registers)
    reg_map = dict(zip(target_sub.registers, builder.registers)) | dict(
        zip(target_sub.aux_registers, target_ar)
    )
    return builder, qubit_map, reg_map


def get_inverted_sub(target_sub: Sub) -> Sub:
    builder, qubit_map, reg_map = _copy_sub_skeleton(target_sub)
    for o, qs, rs in reversed(target_sub.operations):
        qubits = tuple(qubit_map[q] for q in qs)
        regs = tuple(reg_map[r] for r in rs)
        if o.self_inverse:
            io = o
        elif o.unitary:
            io = Inverse(o)
        else:
            io = o
        builder.add_op(io, qubits, regs)
    phase = target_sub.phase
    builder.add_phase(-phase)
    return builder.build()


def inverse_sub_resolver(op: Op, repository: SubRepository) -> Sub | None:
    target_op = op.id.params[0]
    assert isinstance(target_op, Op)

    if target_op.self_inverse:
        return _single_op_sub(target_op)
    target_sub_resolver = repository.find_resolver(target_op)
    if not target_sub_resolver:
        return None
    target_sub = target_sub_resolver(target_op, repository)
    if not target_sub:
        return None

    return get_inverted_sub(target_sub)


def inverse_target_condition(op: AbstractOp) -> SubResolverCondition:
    base_id = op.base_id

    def cond(op_id: Ident) -> bool:
        assert len(op_id.params) > 0
        target_op = op_id.params[0]
        assert isinstance(target_op, Op)
        return target_op.base_id == base_id

    return cond


def _inverse_rotation_resolver_gen(op_factory: OpFactory[float]) -> SubResolver:
    def resolver(op: Op, repository: SubRepository) -> Sub:
        target = op.id.params[0]
        assert isinstance(target, Op)
        angle = target.id.params[0]
        assert isinstance(angle, float)
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(op_factory(-angle), builder.qubits)
        return builder.build()

    return resolver


def _inverse_op_resolver_gen(inverse_op: Op) -> SubResolver:
    def resolver(op: Op, repository: SubRepository) -> Sub:
        target = op.id.params[0]
        assert isinstance(target, Op)
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(inverse_op, builder.qubits)
        return builder.build()

    return resolver


def inverse_controlled_resolver(op: Op, repository: SubRepository) -> Sub:
    target = op.id.params[0]
    assert isinstance(target, Op)
    assert target.base_id == Controlled.base_id
    inner_op = target.id.params[0]
    assert isinstance(inner_op, Op)
    builder = SubBuilder(op.qubit_count, op.reg_count)
    builder.add_op(Controlled(Inverse(inner_op)), builder.qubits)
    return builder.build()


def inverse_multicontrolled_resolver(op: Op, repository: SubRepository) -> Sub:
    target = op.id.params[0]
    assert isinstance(target, Op)
    assert target.base_id == MultiControlled.base_id
    inner_op, control_bits, control_value = target.id.params
    assert isinstance(inner_op, Op)
    builder = SubBuilder(op.qubit_count, op.reg_count)
    builder.add_op(
        MultiControlled(Inverse(inner_op), control_bits, control_value), builder.qubits
    )
    return builder.build()


def inverse_inverse_resolver(op: Op, repository: SubRepository) -> Sub:
    target = op.id.params[0]
    assert isinstance(target, Op)
    assert target.base_id == Inverse.base_id
    inner_op = target.id.params[0]
    assert isinstance(inner_op, Op)
    builder = SubBuilder(op.qubit_count, op.reg_count)
    builder.add_op(inner_op, builder.qubits)
    return builder.build()


_repo = default_repository()
_repo.register_sub_resolver(Inverse, inverse_sub_resolver)

_resolvers: Collection[tuple[AbstractOp, SubResolver]] = [
    (RX, _inverse_rotation_resolver_gen(RX)),
    (RY, _inverse_rotation_resolver_gen(RY)),
    (RZ, _inverse_rotation_resolver_gen(RZ)),
    (Phase, _inverse_rotation_resolver_gen(Phase)),
    (SqrtX, _inverse_op_resolver_gen(SqrtXdag)),
    (SqrtXdag, _inverse_op_resolver_gen(SqrtX)),
    (SqrtY, _inverse_op_resolver_gen(SqrtYdag)),
    (SqrtYdag, _inverse_op_resolver_gen(SqrtY)),
    (S, _inverse_op_resolver_gen(Sdag)),
    (Sdag, _inverse_op_resolver_gen(S)),
    (T, _inverse_op_resolver_gen(Tdag)),
    (Tdag, _inverse_op_resolver_gen(T)),
    (Controlled, inverse_controlled_resolver),
    (MultiControlled, inverse_multicontrolled_resolver),
    (Inverse, inverse_inverse_resolver),
]

for target, resolver in _resolvers:
    _repo.register_sub_resolver(Inverse, resolver, inverse_target_condition(target))
