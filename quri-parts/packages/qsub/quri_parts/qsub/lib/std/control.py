# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Collection, Sequence
from typing import Any

from quri_parts.qsub.lib import std
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
from quri_parts.qsub.register import CTRL_QNAME, QRegSpec
from quri_parts.qsub.resolve import (
    SimpleSubRepository,
    SubRepository,
    SubResolver,
    SubResolverCondition,
    default_repository,
)
from quri_parts.qsub.sub import Sub, SubBuilder

from . import NS
from .cnot import CNOT
from .cz import CZ
from .rotation import RX, RY, RZ, Phase
from .single_clifford import H, S, Sdag, SqrtX, SqrtXdag, SqrtY, SqrtYdag, X, Y, Z
from .swap import SWAP
from .t import T, Tdag
from .toffoli import Toffoli


# Parametric Op definitions
def get_new_ctrl_label(inner_reg_specs: Sequence[QRegSpec]) -> int:
    n_sub_ctrls = sum([int(CTRL_QNAME in rs.name) for rs in inner_reg_specs])
    return n_sub_ctrls


class _Controlled(ParamUnitaryDef[Op]):
    ns = NS
    name = "Controlled"

    def qubit_count_fn(self, target_op: Op) -> int:
        return target_op.qubit_count + 1

    def reg_count_fn(self, target_op: Op) -> int:
        return target_op.reg_count

    def qregs_fn(self, target_op: Op) -> Sequence[QRegSpec]:
        ctrl_label = get_new_ctrl_label(target_op.qregs)
        ctrl_name = CTRL_QNAME if ctrl_label == 0 else f"{CTRL_QNAME}_{ctrl_label}"
        return (QRegSpec(ctrl_name, 1), *target_op.qregs)

    def validate_params(self, target_op: Op) -> None:
        if not target_op.unitary:
            raise ParameterValidationError(f"target_op {target_op} is not unitary")


Controlled: OpFactory[Op] = param_op(_Controlled)


# Sub resolver definitions


def controlled_sub_resolver(op: Op, repository: SubRepository) -> Sub | None:
    target_op = op.id.params[0]
    assert isinstance(target_op, Op)

    target_sub_resolver = repository.find_resolver(target_op)
    if not target_sub_resolver:
        return None
    target_sub = target_sub_resolver(target_op, repository)
    if not target_sub:
        return None

    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    c, *target_q = builder.qubits

    if target_sub.aux_qregs is not None:
        for qr in target_sub.aux_qregs.values():
            builder.add_aux_qreg(qr.name, qr.size)
    target_aq = builder.aux_qubits
    qubit_map = dict(zip(target_sub.qubits, target_q)) | dict(
        zip(target_sub.aux_qubits, target_aq)
    )

    target_ar = tuple(builder.add_aux_register() for _ in target_sub.aux_registers)
    reg_map = dict(zip(target_sub.registers, builder.registers)) | dict(
        zip(target_sub.aux_registers, target_ar)
    )

    for o, qs, rs in target_sub.operations:
        if o.unitary:
            builder.add_op(
                Controlled(o),
                (c, *(qubit_map[q] for q in qs)),
                tuple(reg_map[r] for r in rs),
            )
        else:
            builder.add_op(
                o,
                tuple(qubit_map[q] for q in qs),
                tuple(reg_map[r] for r in rs),
            )

    phase = target_sub.phase % (2 * math.pi)
    if phase == math.pi:
        builder.add_op(Z, (c,))
    elif phase == math.pi / 2:
        builder.add_op(S, (c,))
    elif phase == 3 * math.pi / 2:
        builder.add_op(Sdag, (c,))
    elif phase != 0:
        builder.add_op(Phase(phase), (c,))

    return builder.build()


def control_target_condition(op: AbstractOp) -> SubResolverCondition:
    base_id = op.base_id

    def cond(op_id: Ident) -> bool:
        assert len(op_id.params) > 0
        target_op = op_id.params[0]
        assert isinstance(target_op, Op)
        return target_op.base_id == base_id

    return cond


def controlled_x_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    builder.add_op(CNOT, builder.qubits)
    return builder.build()


def controlled_y_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    c, t = builder.qubits
    builder.add_op(Sdag, (t,))
    builder.add_op(CNOT, (c, t))
    builder.add_op(S, (t,))
    return builder.build()


def controlled_z_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    builder.add_op(CZ, builder.qubits)
    return builder.build()


def controlled_h_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    builder.add_op(RY(-math.pi / 4), (q1,))
    builder.add_op(CZ, (q0, q1))
    builder.add_op(RY(math.pi / 4), (q1,))
    return builder.build()


def _crx(builder: SubBuilder, q0: Qubit, q1: Qubit, angle: float) -> None:
    builder.add_op(H, (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(RZ(-angle / 2), (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(RZ(angle / 2), (q1,))
    builder.add_op(H, (q1,))


def controlled_rx_resolver(op: Op, repository: SubRepository) -> Sub:
    target = op.id.params[0]
    assert isinstance(target, Op)
    angle = target.id.params[0]
    assert isinstance(angle, float)
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    _crx(builder, q0, q1, angle)
    return builder.build()


def _cry(builder: SubBuilder, q0: Qubit, q1: Qubit, angle: float) -> None:
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(RY(-angle / 2), (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(RY(angle / 2), (q1,))


def controlled_ry_resolver(op: Op, repository: SubRepository) -> Sub:
    target = op.id.params[0]
    assert isinstance(target, Op)
    angle = target.id.params[0]
    assert isinstance(angle, float)
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    _cry(builder, q0, q1, angle)
    return builder.build()


def _crz(builder: SubBuilder, q0: Qubit, q1: Qubit, angle: float) -> None:
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(RZ(-angle / 2), (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(RZ(angle / 2), (q1,))


def controlled_rz_resolver(op: Op, repository: SubRepository) -> Sub:
    target = op.id.params[0]
    assert isinstance(target, Op)
    angle = target.id.params[0]
    assert isinstance(angle, float)
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    _crz(builder, q0, q1, angle)
    return builder.build()


def controlled_phase_resolver(op: Op, repository: SubRepository) -> Sub:
    target = op.id.params[0]
    assert isinstance(target, Op)
    angle = target.id.params[0]
    assert isinstance(angle, float)
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    _crz(builder, q0, q1, angle)
    builder.add_op(Phase(angle / 2), (q0,))
    return builder.build()


def controlled_sqrtx_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    _crx(builder, q0, q1, math.pi / 2)
    builder.add_op(Phase(math.pi / 4), (q0,))
    return builder.build()


def controlled_sqrtxdag_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    _crx(builder, q0, q1, -math.pi / 2)
    builder.add_op(Phase(-math.pi / 4), (q0,))
    return builder.build()


def controlled_sqrty_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    _cry(builder, q0, q1, math.pi / 2)
    builder.add_op(Phase(math.pi / 4), (q0,))
    return builder.build()


def controlled_sqrtydag_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    _cry(builder, q0, q1, -math.pi / 2)
    builder.add_op(Phase(-math.pi / 4), (q0,))
    return builder.build()


def controlled_s_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(Tdag, (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(T, (q0,))
    builder.add_op(T, (q1,))
    return builder.build()


def controlled_sdag_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(T, (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(Tdag, (q0,))
    builder.add_op(Tdag, (q1,))
    return builder.build()


def controlled_t_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(Phase(-math.pi / 8), (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(Phase(math.pi / 8), (q0,))
    builder.add_op(Phase(math.pi / 8), (q1,))
    return builder.build()


def controlled_tdag_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1 = builder.qubits
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(Phase(math.pi / 8), (q1,))
    builder.add_op(CNOT, (q0, q1))
    builder.add_op(Phase(-math.pi / 8), (q0,))
    builder.add_op(Phase(-math.pi / 8), (q1,))
    return builder.build()


def controlled_cnot_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    builder.add_op(Toffoli, builder.qubits)
    return builder.build()


def controlled_cz_resolver(op: Op, repository: SubRepository) -> Sub:
    from .multi_control import MultiControlled

    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    builder.add_op(MultiControlled(Z, 2, 0b11), builder.qubits)
    return builder.build()


def controlled_swap_resolver(op: Op, repository: SubRepository) -> Sub:
    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    q0, q1, q2 = builder.qubits
    builder.add_op(CNOT, (q2, q1))
    builder.add_op(Toffoli, (q0, q1, q2))
    builder.add_op(CNOT, (q2, q1))
    return builder.build()


def controlled_toffoli_resolver(op: Op, repository: SubRepository) -> Sub:
    from .multi_control import MultiControlled

    builder = SubBuilder(op.qubit_count, op.reg_count, op.qregs)
    builder.add_op(MultiControlled(X, 3, 0b111), builder.qubits)
    return builder.build()


def _get_ctrl_inverse_resolver(
    ctrl_resolver: SubResolver, repository: SubRepository
) -> SubResolver:
    from quri_parts.qsub.lib.std.inverse import get_inverted_sub

    def get_inner_op(op: Op) -> Op:
        params = op.id.params
        assert len(params) == 1
        inv_op = params[0]
        assert isinstance(inv_op, Op)
        inner_params = inv_op.id.params
        assert len(inner_params) == 1
        inner_op = inner_params[0]
        assert isinstance(inner_op, Op)
        return inner_op

    class ControlInvSubResolver(SubResolver):
        def __call__(self, ctrl_inv_op: Op, repository: SubRepository) -> Sub | None:
            # ctrl_inv_op is supposed to be Controlled(Inverse(op))
            inner_op = get_inner_op(ctrl_inv_op)
            ctrl_sub = ctrl_resolver(Controlled(inner_op), repository)
            assert ctrl_sub is not None
            inv_ctrl_sub = get_inverted_sub(ctrl_sub)
            return inv_ctrl_sub

    return ControlInvSubResolver()


def _get_inv_target_condition(op: AbstractOp) -> SubResolverCondition:
    inner_base_id = op.base_id

    def cond(op_id: Ident) -> bool:
        assert len(op_id.params) > 0
        outer_op = op_id.params[0]
        assert isinstance(outer_op, Op)
        if outer_op.base_id != std.Inverse.base_id:
            return False
        assert len(outer_op.id.params) > 0
        inner_op = outer_op.id.params[0]
        assert isinstance(inner_op, Op)
        return inner_op.base_id == inner_base_id

    return cond


def register_controlled_resolver(
    sub_repository: SimpleSubRepository,
    control_resolver: SubResolver,
    op: Op | OpFactory[Any],
) -> None:
    """"""
    sub_repository.register_sub_resolver(
        Controlled, control_resolver, control_target_condition(op)
    )
    inv_ctrl_resolver = _get_ctrl_inverse_resolver(control_resolver, sub_repository)
    cond = _get_inv_target_condition(op)
    sub_repository.register_sub_resolver(Controlled, inv_ctrl_resolver, cond)


_repo = default_repository()
_repo.register_sub_resolver(Controlled, controlled_sub_resolver)

_resolvers: Collection[tuple[AbstractOp, SubResolver]] = [
    (X, controlled_x_resolver),
    (Y, controlled_y_resolver),
    (Z, controlled_z_resolver),
    (H, controlled_h_resolver),
    (RX, controlled_rx_resolver),
    (RY, controlled_ry_resolver),
    (RZ, controlled_rz_resolver),
    (Phase, controlled_phase_resolver),
    (SqrtX, controlled_sqrtx_resolver),
    (SqrtXdag, controlled_sqrtxdag_resolver),
    (SqrtY, controlled_sqrty_resolver),
    (SqrtYdag, controlled_sqrtydag_resolver),
    (S, controlled_s_resolver),
    (Sdag, controlled_sdag_resolver),
    (T, controlled_t_resolver),
    (Tdag, controlled_tdag_resolver),
    (CNOT, controlled_cnot_resolver),
    (CZ, controlled_cz_resolver),
    (SWAP, controlled_swap_resolver),
    (Toffoli, controlled_toffoli_resolver),
]

for target, resolver in _resolvers:
    _repo.register_sub_resolver(Controlled, resolver, control_target_condition(target))
