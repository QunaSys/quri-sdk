# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Optional

from quri_parts.qsub.op import (
    Op,
    OpFactory,
    ParameterValidationError,
    ParamUnitaryDef,
    param_op,
)
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.resolve import SubRepository, default_repository
from quri_parts.qsub.sub import Sub, SubBuilder

from . import NS
from .control import Controlled, control_target_condition
from .logic import scoped_and, scoped_and_clifford_t, scoped_and_single_toffoli
from .single_clifford import X

# Parametric Op definitions


class _MultiControlled(ParamUnitaryDef[Op, int, int]):
    ns = NS
    name = "MultiControlled"

    def qubit_count_fn(
        self, target_op: Op, control_bits: int, control_value: int
    ) -> int:
        return target_op.qubit_count + control_bits

    def reg_count_fn(self, target_op: Op, control_bits: int, control_value: int) -> int:
        return target_op.reg_count

    def validate_params(
        self, target_op: Op, control_bits: int, control_value: int
    ) -> None:
        if not target_op.unitary:
            raise ParameterValidationError(f"target_op {target_op} is not unitary")
        if not control_bits >= 1:
            raise ParameterValidationError(
                f"control_bits should be a positive integer but {control_bits}"
            )
        if not (control_value >= 0 and control_value < 2**control_bits):
            raise ParameterValidationError(
                f"control_value should be a {control_bits}-bits integer but"
                f"{control_value}"
            )


MultiControlled: OpFactory[Op, int, int] = param_op(_MultiControlled)


def _multi_controlled_sub(
    op: Op,
    control_bits: int,
    control_value: Optional[int],
    s_and: Callable[[SubBuilder, Qubit, Qubit], AbstractContextManager[Qubit]],
) -> Sub:
    builder = SubBuilder(op.qubit_count + control_bits)
    qubits = builder.qubits

    if control_value is None:
        control_value = (1 << control_bits) - 1

    if not control_bits >= 1:
        raise ValueError(
            f"control_bits should be a positive integer but specified: "
            f"{control_bits}"
        )

    neg_qubits = [i for i in range(control_bits) if ((control_value >> i) & 1) == 0]

    def add_neg() -> None:
        for i in neg_qubits:
            builder.add_op(X, (qubits[i],))

    if control_bits == 1:
        add_neg()
        builder.add_op(Controlled(op), (qubits[0], qubits[1]))
        add_neg()
        return builder.build()

    def recursive_and(i: int, a: Optional[Qubit] = None) -> None:
        if i == control_bits - 1:
            assert a
            builder.add_op(Controlled(op), (a, *qubits[control_bits:]))
        else:
            if i == 0:
                i0 = qubits[0]
            else:
                assert a
                i0 = a
            i1 = qubits[i + 1]
            with s_and(builder, i0, i1) as a:
                recursive_and(i + 1, a)

    add_neg()
    recursive_and(0)
    add_neg()

    return builder.build()


def MultiControlledSub(
    op: Op, control_bits: int, control_value: Optional[int] = None
) -> Sub:
    return _multi_controlled_sub(op, control_bits, control_value, scoped_and)


def MultiControlledCliffordTSub(op: Op, control_bits: int, control_value: int) -> Sub:
    return _multi_controlled_sub(op, control_bits, control_value, scoped_and_clifford_t)


def MultiControlledSingleToffoliSub(
    op: Op, control_bits: int, control_value: int
) -> Sub:
    return _multi_controlled_sub(
        op, control_bits, control_value, scoped_and_single_toffoli
    )


def controlled_multicontrolled_resolver(op: Op, repository: SubRepository) -> Sub:
    target_op = op.id.params[0]
    assert isinstance(target_op, Op)
    inner_op, control_bits, control_value = target_op.id.params
    assert isinstance(inner_op, Op)
    assert isinstance(control_bits, int)
    assert isinstance(control_value, int)
    control_bits += 1
    control_value = (control_value << 1) + 1
    builder = SubBuilder(op.qubit_count, op.reg_count)
    builder.add_op(
        MultiControlled(inner_op, control_bits, control_value), builder.qubits
    )
    return builder.build()


# Register sub resolvers
_repo = default_repository()
_repo.register_sub(MultiControlled, MultiControlledSub)
_repo.register_sub_resolver(
    Controlled,
    controlled_multicontrolled_resolver,
    control_target_condition(MultiControlled),
)
