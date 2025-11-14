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
from contextlib import AbstractContextManager
from typing import Callable

from quri_parts.qsub.op import (
    BaseIdent,
    Op,
    OpFactory,
    ParameterValidationError,
    ParamUnitaryDef,
    param_op,
)
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.resolve import SubRepository, SubResolver, default_repository
from quri_parts.qsub.sub import Sub, SubBuilder

from . import NS
from .cnot import CNOT
from .control import Controlled
from .cz import CZ
from .logic import scoped_and
from .multi_control import MultiControlled, _multi_controlled_sub
from .rotation import RX, RY, RZ, Phase
from .single_clifford import (
    H,
    Identity,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    X,
    Y,
    Z,
)
from .swap import SWAP
from .t import T, Tdag
from .toffoli import Toffoli


class _MCBase(ParamUnitaryDef[int]):
    ns = NS

    def qubit_count_fn(self, control_bits: int) -> int:
        return control_bits + 1

    def reg_count_fn(self, _control_bits: int) -> int:
        return 0

    def validate_params(self, control_bits: int) -> None:
        if not control_bits >= 1:
            raise ParameterValidationError(
                f"control_bits should be a positive integer but {control_bits}"
            )


class _MCRotationBase(ParamUnitaryDef[int, float]):
    ns = NS

    def qubit_count_fn(self, control_bits: int, _angle: float) -> int:
        return control_bits + 1

    def reg_count_fn(self, _control_bits: int, _angle: float) -> int:
        return 0

    def validate_params(self, control_bits: int, _: float) -> None:
        if not control_bits >= 1:
            raise ParameterValidationError(
                f"control_bits should be a positive integer but {control_bits}"
            )


# Multi-controlled gates with variable control bits
class _MCX(_MCBase):
    name = "MCX"


class _MCY(_MCBase):
    name = "MCY"


class _MCZ(_MCBase):
    name = "MCZ"


class _MCS(_MCBase):
    name = "MCS"


class _MCSdag(_MCBase):
    name = "MCSdag"


class _MCT(_MCBase):
    name = "MCT"


class _MCTdag(_MCBase):
    name = "MCTdag"


class _MCSqrtX(_MCBase):
    name = "MCSqrtX"


class _MCSqrtXdag(_MCBase):
    name = "MCSqrtXdag"


class _MCSqrtY(_MCBase):
    name = "MCSqrtY"


class _MCSqrtYdag(_MCBase):
    name = "MCSqrtYdag"


class _MCH(_MCBase):
    name = "MCH"


class _MCRZ(_MCRotationBase):
    name = "MCRZ"


class _MCRX(_MCRotationBase):
    name = "MCRX"


class _MCRY(_MCRotationBase):
    name = "MCRY"


class _MCPhase(_MCRotationBase):
    name = "MCPhase"


#: Multi-controlled X gate. Applies X gate to target qubit when all control
#: qubits are in |1⟩ state.
MCX: OpFactory[int] = param_op(_MCX)
#: Multi-controlled Y gate. Applies Y gate to target qubit when all control
#: qubits are in |1⟩ state.
MCY: OpFactory[int] = param_op(_MCY)
#: Multi-controlled Z gate. Applies Z gate to target qubit when all control
#: qubits are in |1⟩ state.
MCZ: OpFactory[int] = param_op(_MCZ)
#: Multi-controlled S gate. Applies S gate to target qubit when all control
#: qubits are in |1⟩ state.
MCS: OpFactory[int] = param_op(_MCS)
#: Multi-controlled S† gate. Applies S† gate to target qubit when all
#: control qubits are in |1⟩ state.
MCSdag: OpFactory[int] = param_op(_MCSdag)
#: Multi-controlled T gate. Applies T gate to target qubit when all control
#: qubits are in |1⟩ state.
MCT: OpFactory[int] = param_op(_MCT)
#: Multi-controlled T† gate. Applies T† gate to target qubit when all
#: control qubits are in |1⟩ state.
MCTdag: OpFactory[int] = param_op(_MCTdag)
#: Multi-controlled √X gate. Applies √X gate to target qubit when all
#: control qubits are in |1⟩ state.
MCSqrtX: OpFactory[int] = param_op(_MCSqrtX)
#: Multi-controlled √X† gate. Applies √X† gate to target qubit when all
#: control qubits are in |1⟩ state.
MCSqrtXdag: OpFactory[int] = param_op(_MCSqrtXdag)
#: Multi-controlled √Y gate. Applies √Y gate to target qubit when all
#: control qubits are in |1⟩ state.
MCSqrtY: OpFactory[int] = param_op(_MCSqrtY)
#: Multi-controlled √Y† gate. Applies √Y† gate to target qubit when all
#: control qubits are in |1⟩ state.
MCSqrtYdag: OpFactory[int] = param_op(_MCSqrtYdag)
#: Multi-controlled Hadamard gate. Applies H gate to target qubit when all
#: control qubits are in |1⟩ state.
MCH: OpFactory[int] = param_op(_MCH)
#: Multi-controlled RZ rotation gate. Applies RZ(angle) to target qubit when
#: all control qubits are in |1⟩ state.
MCRZ: OpFactory[int, float] = param_op(_MCRZ)
#: Multi-controlled RX rotation gate. Applies RX(angle) to target qubit when
#: all control qubits are in |1⟩ state.
MCRX: OpFactory[int, float] = param_op(_MCRX)
#: Multi-controlled RY rotation gate. Applies RY(angle) to target qubit when
#: all control qubits are in |1⟩ state.
MCRY: OpFactory[int, float] = param_op(_MCRY)
#: Multi-controlled Phase gate. Applies Phase(angle) to target qubit when
#: all control qubits are in |1⟩ state.
MCPhase: OpFactory[int, float] = param_op(_MCPhase)


# Sub resolver definitions
_mc_gate_mapping: dict[BaseIdent, OpFactory[[int]]] = {
    X.base_id: MCX,
    Y.base_id: MCY,
    Z.base_id: MCZ,
    H.base_id: MCH,
    S.base_id: MCS,
    Sdag.base_id: MCSdag,
    T.base_id: MCT,
    Tdag.base_id: MCTdag,
    SqrtX.base_id: MCSqrtX,
    SqrtXdag.base_id: MCSqrtXdag,
    SqrtY.base_id: MCSqrtY,
    SqrtYdag.base_id: MCSqrtYdag,
}

_mc_gate_mapping_param: dict[BaseIdent, OpFactory[[int, float]]] = {
    RX.base_id: MCRX,
    RY.base_id: MCRY,
    RZ.base_id: MCRZ,
    Phase.base_id: MCPhase,
}


def MultiControlledNamedMCGatesSub(
    target_op: Op, control_bits: int, control_value: int
) -> Sub | None:
    """Resolves MultiControlled operations using named multi-controlled gates.

    Maps known target operations to their corresponding named MC gate variants
    (MCX, MCY, MCZ, MCS, etc.) when available. For unknown operations, returns
    None to allow the system to fall back to the standard MultiControlledSub resolver.

    Please use :func:`generate_multicontrolled_to_mc_sub_resolver()` instead, which
    also resolves MultiControlled gates, generating known MC? gates and fallback
    with unknown ops.
    """
    builder = SubBuilder(target_op.qubit_count + control_bits, target_op.reg_count)
    qubits = builder.qubits

    # Handle negation using add_neg pattern
    neg_qubits = [i for i in range(control_bits) if ((control_value >> i) & 1) == 0]

    def add_neg() -> None:
        for i in neg_qubits:
            builder.add_op(X, (qubits[i],))

    # Check if target_op is in mc_gate_mapping and use appropriate
    # multi-controlled gate
    if (
        target_op.base_id in _mc_gate_mapping
        or target_op.base_id in _mc_gate_mapping_param
    ):
        # First apply negations if needed
        add_neg()

        # Handle parametrized gates (Phase, RX, RY, RZ)
        if target_op.base_id in _mc_gate_mapping_param:
            # Extract angle parameter from target_op
            mc_gate_factory = _mc_gate_mapping_param[target_op.base_id]
            angle = target_op.id.params[0]
            assert isinstance(angle, float)
            builder.add_op(mc_gate_factory(control_bits, angle), qubits)
        else:
            # Non-parametrized gates
            mc_gate_factory2 = _mc_gate_mapping[target_op.base_id]
            builder.add_op(mc_gate_factory2(control_bits), qubits)

        add_neg()
    elif target_op.base_id == Toffoli.base_id:
        builder.add_op(
            MultiControlled(
                X, control_bits + 2, control_value + ((2 + 1) << control_bits)
            ),
            qubits,
        )
    elif target_op.base_id == CNOT.base_id:
        if control_bits == 1:
            add_neg()
            builder.add_op(Toffoli, qubits)
            add_neg()
        else:
            builder.add_op(
                MultiControlled(
                    X, control_bits + 1, control_value + (1 << control_bits)
                ),
                qubits,
            )
    elif target_op.base_id == CZ.base_id:
        builder.add_op(
            MultiControlled(Z, control_bits + 1, control_value + (1 << control_bits)),
            qubits,
        )
    elif target_op.base_id == MultiControlled.base_id:
        mc_target_op, mc_control_bits, mc_control_value = target_op.id.params
        assert isinstance(mc_target_op, Op)
        assert isinstance(mc_control_bits, int)
        assert isinstance(mc_control_value, int)
        builder.add_op(
            MultiControlled(
                mc_target_op,
                control_bits + mc_control_bits,
                control_value + (mc_control_value << control_bits),
            ),
            qubits,
        )
    elif target_op.base_id == Controlled.base_id:
        assert len(target_op.id.params) == 1
        c_op = target_op.id.params[0]
        assert isinstance(c_op, Op)
        builder.add_op(
            MultiControlled(
                c_op,
                control_bits + 1,
                control_value + (1 << control_bits),
            ),
            qubits,
        )
    elif target_op.base_id == SWAP.base_id:
        assert len(qubits) == control_bits + 2
        builder.add_op(CNOT, [qubits[-1], qubits[-2]])
        builder.add_op(
            MultiControlled(
                CNOT,
                control_bits,
                control_value,
            ),
            qubits,
        )
        builder.add_op(CNOT, [qubits[-1], qubits[-2]])
    elif target_op.base_id == Identity.base_id:
        for q in qubits:
            builder.add_op(Identity, [q])
    else:
        return None
    return builder.build()


def generate_multicontrolled_to_mc_sub_resolver(
    s_and: Callable[
        [SubBuilder, Qubit, Qubit], AbstractContextManager[Qubit]
    ] = scoped_and  # type: ignore
) -> SubResolver:
    """Generates a resolver that expands MultiControlled to named MC operations
    when possible.

    This resolver is useful when you expect named MC gates as final
    primitives, especially
    when converting circuits to SimulatorBasicSet. In other cases, it may emit
    redundant Toffoli (or s_and) chains during the compilation process.

    From the end-user perspective, all MultiControlled operations work: known
    operations use optimized named gates, while unknown operations use the standard
    decomposition.

    Algorithm:
    1. Evaluate MultiControlledNamedMCGatesSub to try converting the MultiControlled
       operation to a named MC gate.
    2. If that fails, try to resolve the target operation. If successful, wrap each
       operation in the sub with MultiControlled.
    3. If that also fails, decompose the MultiControlled using Toffoli or s_and
       (default behavior).

    During expansion, adds MultiControlled Phase gates according to the global
    phase of the operation.

    Example:
        >>> new_repo = default_repository().copy()
        >>> new_repo.register_sub_resolver(
        ...     MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
        ... )
    """

    def resolver(op: Op, repository: SubRepository) -> Sub | None:
        target_op, control_bits, control_value = op.id.params
        assert isinstance(target_op, Op)
        assert isinstance(control_bits, int)
        assert isinstance(control_value, int)

        named_sub = MultiControlledNamedMCGatesSub(
            target_op, control_bits, control_value
        )
        if named_sub is not None:
            return named_sub

        target_sub_resolver = repository.find_resolver(target_op)
        target_sub = None
        if target_sub_resolver is not None:
            target_sub = target_sub_resolver(target_op, repository)

        if target_sub is None:
            return _multi_controlled_sub(target_op, control_bits, control_value, s_and)

        builder = SubBuilder(op.qubit_count, op.reg_count)
        control_q = builder.qubits[:control_bits]
        target_q = builder.qubits[control_bits:]

        target_aq = tuple(builder.add_aux_qubit() for _ in target_sub.aux_qubits)
        qubit_map = dict(zip(target_sub.qubits, target_q)) | dict(
            zip(target_sub.aux_qubits, target_aq)
        )

        target_ar = tuple(builder.add_aux_register() for _ in target_sub.aux_registers)
        reg_map = dict(zip(target_sub.registers, builder.registers)) | dict(
            zip(target_sub.aux_registers, target_ar)
        )

        for o, qs, rs in target_sub.operations:
            if not o.unitary:
                raise ValueError(f"Unsupported operation, {o} in multi-controlled sub")

            builder.add_op(
                MultiControlled(o, control_bits, control_value),
                (*control_q, *(qubit_map[q] for q in qs)),
                tuple(reg_map[r] for r in rs),
            )

        if target_sub.phase != 0:
            phase = target_sub.phase % (2 * math.pi)
            if (control_value & (1 << (control_bits - 1))) == 0:
                # use diag(e^i(phase), 1)  instead of diag(1, e^i(phase))
                phase = (-phase) % (2 * math.pi)
                builder.add_phase(phase)
            if phase == math.pi:
                phase_op = Z
            elif phase == math.pi / 2:
                phase_op = S
            elif phase == 3 * math.pi / 2:
                phase_op = Sdag
            else:
                phase_op = Phase(phase)

            if control_bits == 1:
                builder.add_op(phase_op, qubits=control_q)
            elif control_bits > 1:
                builder.add_op(
                    MultiControlled(
                        phase_op,
                        control_bits - 1,
                        control_value & ((1 << (control_bits - 1)) - 1),
                    ),
                    qubits=control_q,
                )
            else:
                raise ValueError("unreachable")

        return builder.build()

    return resolver


# MCX, MCY, MCZ resolvers using recursive decomposition
def mcx_resolver(op: Op, repository: SubRepository) -> Sub:
    from .multi_control import MultiControlledSub

    control_bits = op.id.params[0]
    assert isinstance(control_bits, int)

    if control_bits == 1:
        # Single control = CNOT
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(CNOT, builder.qubits)
        return builder.build()
    elif control_bits == 2:
        # Double control = Toffoli
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Toffoli, builder.qubits)
        return builder.build()
    else:
        # General multi-controlled X using recursive decomposition
        return MultiControlledSub(X, control_bits)


def mcy_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits = op.id.params[0]
    assert isinstance(control_bits, int)

    if control_bits == 1:
        # Single controlled Y
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(Y), builder.qubits)
        return builder.build()
    elif control_bits == 2:
        # Double controlled Y = Sdag + Toffoli + S
        builder = SubBuilder(op.qubit_count, op.reg_count)
        qubits = builder.qubits
        target = qubits[-1]
        builder.add_op(Sdag, (target,))
        builder.add_op(Toffoli, qubits)
        builder.add_op(S, (target,))
        return builder.build()
    else:
        # General multi-controlled Y using recursive decomposition
        return MultiControlledSub(Y, control_bits)


def mcz_resolver(op: Op, repository: SubRepository) -> Sub:
    from .multi_control import MultiControlledSub

    control_bits = op.id.params[0]
    assert isinstance(control_bits, int)

    if control_bits == 1:
        # Single controlled Z = CZ
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(CZ, builder.qubits)
        return builder.build()
    elif control_bits == 2:
        # Double controlled Z = H + Toffoli + H
        builder = SubBuilder(op.qubit_count, op.reg_count)
        qubits = builder.qubits
        target = qubits[-1]
        builder.add_op(H, (target,))
        builder.add_op(Toffoli, qubits)
        builder.add_op(H, (target,))
        return builder.build()
    else:
        # General multi-controlled Z using recursive decomposition
        return MultiControlledSub(Z, control_bits)


# MCS, MCSdag, MCT, MCTdag, MCSqrtX, MCSqrtXdag, MCSqrtY, MCSqrtYdag resolvers
def mcs_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits = op.id.params[0]
    assert isinstance(control_bits, int)

    if control_bits == 1:
        # Single controlled S
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(S), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled S using recursive decomposition
        return MultiControlledSub(S, control_bits)


def mcsdag_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits = op.id.params[0]
    assert isinstance(control_bits, int)

    if control_bits == 1:
        # Single controlled Sdag
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(Sdag), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled Sdag using recursive decomposition
        return MultiControlledSub(Sdag, control_bits)


def mct_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits = op.id.params[0]
    assert isinstance(control_bits, int)

    if control_bits == 1:
        # Single controlled T
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(T), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled T using recursive decomposition
        return MultiControlledSub(T, control_bits)


def mctdag_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits = op.id.params[0]
    assert isinstance(control_bits, int)

    if control_bits == 1:
        # Single controlled Tdag
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(Tdag), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled Tdag using recursive decomposition
        return MultiControlledSub(Tdag, control_bits)


def mcsqrtx_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits = op.id.params[0]
    assert isinstance(control_bits, int)

    if control_bits == 1:
        # Single controlled SqrtX
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(SqrtX), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled SqrtX using recursive decomposition
        return MultiControlledSub(SqrtX, control_bits)


def mcsqrtxdag_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits = op.id.params[0]
    assert isinstance(control_bits, int)

    if control_bits == 1:
        # Single controlled SqrtXdag
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(SqrtXdag), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled SqrtXdag using recursive decomposition
        return MultiControlledSub(SqrtXdag, control_bits)


def mcsqrty_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits = op.id.params[0]
    assert isinstance(control_bits, int)

    if control_bits == 1:
        # Single controlled SqrtY
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(SqrtY), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled SqrtY using recursive decomposition
        return MultiControlledSub(SqrtY, control_bits)


def mcsqrtydag_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits = op.id.params[0]
    assert isinstance(control_bits, int)

    if control_bits == 1:
        # Single controlled SqrtYdag
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(SqrtYdag), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled SqrtYdag using recursive decomposition
        return MultiControlledSub(SqrtYdag, control_bits)


def mch_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits = op.id.params[0]
    assert isinstance(control_bits, int)

    if control_bits == 1:
        # Single controlled H
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(H), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled H using recursive decomposition
        return MultiControlledSub(H, control_bits)


def mcphase_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits, angle = op.id.params
    assert isinstance(control_bits, int)
    assert isinstance(angle, float)

    if control_bits == 1:
        # Single controlled Phase
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(Phase(angle)), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled Phase using recursive decomposition
        return MultiControlledSub(Phase(angle), control_bits)


def mcrz_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits, angle = op.id.params
    assert isinstance(control_bits, int)
    assert isinstance(angle, float)

    if control_bits == 1:
        # Single controlled RZ
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(RZ(angle)), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled RZ using recursive decomposition
        return MultiControlledSub(RZ(angle), control_bits)


def mcrx_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits, angle = op.id.params
    assert isinstance(control_bits, int)
    assert isinstance(angle, float)

    if control_bits == 1:
        # Single controlled RX
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(RX(angle)), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled RX using recursive decomposition
        return MultiControlledSub(RX(angle), control_bits)


def mcry_resolver(op: Op, repository: SubRepository) -> Sub:
    from .control import Controlled
    from .multi_control import MultiControlledSub

    control_bits, angle = op.id.params
    assert isinstance(control_bits, int)
    assert isinstance(angle, float)

    if control_bits == 1:
        # Single controlled RY
        builder = SubBuilder(op.qubit_count, op.reg_count)
        builder.add_op(Controlled(RY(angle)), builder.qubits)
        return builder.build()
    else:
        # General multi-controlled RY using recursive decomposition
        return MultiControlledSub(RY(angle), control_bits)


# Register sub resolvers
_repo = default_repository()

# Register MCX, MCY, MCZ resolvers
_repo.register_sub_resolver(MCX, mcx_resolver)
_repo.register_sub_resolver(MCY, mcy_resolver)
_repo.register_sub_resolver(MCZ, mcz_resolver)

# Register MCS, MCSdag, MCT, MCTdag, MCSqrtX, MCSqrtXdag,
# MCSqrtY, MCSqrtYdag, MCH resolvers
_repo.register_sub_resolver(MCS, mcs_resolver)
_repo.register_sub_resolver(MCSdag, mcsdag_resolver)
_repo.register_sub_resolver(MCT, mct_resolver)
_repo.register_sub_resolver(MCTdag, mctdag_resolver)
_repo.register_sub_resolver(MCSqrtX, mcsqrtx_resolver)
_repo.register_sub_resolver(MCSqrtXdag, mcsqrtxdag_resolver)
_repo.register_sub_resolver(MCSqrtY, mcsqrty_resolver)
_repo.register_sub_resolver(MCSqrtYdag, mcsqrtydag_resolver)
_repo.register_sub_resolver(MCH, mch_resolver)

# Register MCPhase, MCRZ, MCRX, MCRY, MCU1 resolvers
_repo.register_sub_resolver(MCRZ, mcrz_resolver)
_repo.register_sub_resolver(MCRX, mcrx_resolver)
_repo.register_sub_resolver(MCRY, mcry_resolver)
_repo.register_sub_resolver(MCPhase, mcphase_resolver)
