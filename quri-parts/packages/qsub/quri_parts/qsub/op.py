# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    NamedTuple,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    TypeAlias,
)

from .namespace import DEFAULT, NameSpace
from .param import ArrayRef
from .register import QRegSpec, check_register_appear_once, get_default_qreg_sequence

BaseIdent: TypeAlias = tuple[NameSpace, str]
# Param should contain str but adding it makes mypy fail with segmentation fault
Param: TypeAlias = "int | float | Ident | Op | ArrayRef[Any] | tuple[Param, ...]"
Params = ParamSpec("Params")


def _param_short_str(p: Param) -> str:
    if isinstance(p, Op):
        return str(p.id)
    else:
        return str(p)


class Ident(NamedTuple):
    ns: NameSpace
    local_name: str
    params: tuple[Param, ...] = ()

    @property
    def base(self) -> BaseIdent:
        return (self.ns, self.local_name)

    def to_str(self, full: bool = True, param_truncate: int = 0) -> str:
        if full:
            ns = str(self.ns)
            s = ".".join((ns, self.local_name))
        else:
            s = self.local_name
        if self.params:
            params_str = tuple(_param_short_str(p) for p in self.params)
            joined = ", ".join(params_str)
            if param_truncate > 0 and len(joined) > param_truncate:
                joined = joined[:param_truncate] + "..."
            return f"{s}<{joined}>"
        else:
            return s

    def __str__(self) -> str:
        return self.to_str(full=True)


class AbstractOp(Protocol):
    @property
    def base_id(self) -> BaseIdent:
        ...


@dataclass(frozen=True)
class Op:
    id: Ident
    qubit_count: int
    qregs: Sequence[QRegSpec]
    reg_count: int = 0
    unitary: bool = True
    self_inverse: bool = False

    def __post_init__(self) -> None:
        self._check_qubit_count()
        check_register_appear_once(self.qregs)

    @staticmethod
    def from_qregs(
        id: Ident,
        qregs: Sequence[QRegSpec],
        reg_count: int = 0,
        unitary: bool = True,
        self_inverse: bool = False,
    ) -> "Op":
        qubit_count = sum([qr.qubit_count for qr in qregs])
        return Op(id, qubit_count, qregs, reg_count, unitary, self_inverse)

    @staticmethod
    def from_qubit_count(
        id: Ident,
        qubit_count: int,
        reg_count: int = 0,
        unitary: bool = True,
        self_inverse: bool = False,
    ) -> "Op":
        qregs = get_default_qreg_sequence(qubit_count)
        return Op(id, qubit_count, qregs, reg_count, unitary, self_inverse)

    @property
    def base_id(self) -> BaseIdent:
        return self.id.base

    def __str__(self) -> str:
        qreg_str = ", ".join(f"{qr.name}: {qr.qubit_count}" for qr in self.qregs)
        arg_str = ", ".join(
            [
                f"qubits={self.qubit_count}",
                f"qregs=<{qreg_str}>",
                f"registers={self.reg_count}",
                f"unitary={self.unitary}",
                f"self_inv={self.self_inverse}",
            ]
        )
        return f"{str(self.id)}({arg_str})"

    def _check_qubit_count(self) -> None:
        actual_qubit_cnts = sum([r.qubit_count for r in self.qregs])
        assert (
            actual_qubit_cnts == self.qubit_count
        ), f"Expecting {self.qubit_count} qubits, but got {actual_qubit_cnts} qubits."


class OpFactory(Protocol[Params]):
    @property
    def base_id(self) -> BaseIdent:
        ...

    def __call__(self, *params: Params.args) -> Op:
        ...


class OpDef:
    ns: NameSpace = DEFAULT
    name: str
    params: tuple[Param, ...] = ()
    qubit_count: int
    reg_count: int = 0
    unitary: bool = True
    self_inverse: bool = False
    qregs: Sequence[QRegSpec] | None = None

    @classmethod
    def get_qregs(cls) -> Sequence[QRegSpec]:
        if cls.qregs is not None:
            return cls.qregs
        return get_default_qreg_sequence(cls.qubit_count)


def op(op_def: type[OpDef]) -> Op:
    ident = Ident(ns=op_def.ns, local_name=op_def.name, params=op_def.params)
    qubit_count = op_def.qubit_count
    reg_count = op_def.reg_count
    unitary = op_def.unitary
    self_inverse = op_def.self_inverse
    return Op(
        id=ident,
        qubit_count=qubit_count,
        reg_count=reg_count,
        unitary=unitary,
        self_inverse=self_inverse,
        qregs=op_def.get_qregs(),
    )


class UnitaryDef(OpDef):
    unitary = True


class NonUnitaryDef(OpDef):
    unitary = False


class ParametricMixin(Generic[Params]):
    qubit_count: Optional[int]
    reg_count: Optional[int] = 0
    self_inverse: bool = False
    qregs: Sequence[QRegSpec] | None = None

    def qubit_count_fn(self, *params: Params.args, **_: Params.kwargs) -> int:
        if self.qubit_count is not None:
            return self.qubit_count
        else:
            raise ValueError("qubit_count attribute is not set")

    def reg_count_fn(self, *params: Params.args, **_: Params.kwargs) -> int:
        if self.reg_count is not None:
            return self.reg_count
        else:
            raise ValueError("reg_count attribute is not set")

    def qregs_fn(self, *params: Params.args, **_: Params.kwargs) -> Sequence[QRegSpec]:
        if self.qregs is not None:
            return self.qregs
        else:
            return get_default_qreg_sequence(self.qubit_count_fn(*params))

    def self_inverse_fn(self, *params: Params.args, **_: Params.kwargs) -> bool:
        return self.self_inverse


class ParameterValidationError(Exception):
    ...


class ParamOpDef(ParametricMixin[Params]):
    ns: NameSpace = DEFAULT
    name: str
    unitary: bool = True

    def validate_params(self, *params: Params.args, **_: Params.kwargs) -> None:
        ...


class ParamUnitaryDef(ParamOpDef[Params]):
    unitary = True


class ParamNonUnitaryDef(ParamOpDef[Params]):
    unitary = False


@dataclass
class _ParamOpFactory(OpFactory[Params]):
    op_def: ParamOpDef[Params]

    @property
    def base_id(self) -> BaseIdent:
        return (self.op_def.ns, self.op_def.name)

    def __call__(self, *params: Params.args) -> Op:
        self.op_def.validate_params(*params)
        ident = Ident(self.op_def.ns, self.op_def.name, params=params)
        qubit_count = self.op_def.qubit_count_fn(*params)
        reg_count = self.op_def.reg_count_fn(*params)
        unitary = self.op_def.unitary
        self_inverse = self.op_def.self_inverse_fn(*params)
        qregs = self.op_def.qregs_fn(*params)
        return Op(
            id=ident,
            qubit_count=qubit_count,
            reg_count=reg_count,
            unitary=unitary,
            self_inverse=self_inverse,
            qregs=qregs,
        )


def param_op(op_def: type[ParamOpDef[Params]]) -> OpFactory[Params]:
    return _ParamOpFactory(op_def())


@dataclass(frozen=True)
class SimpleParamOp(ParametricMixin[Params]):
    """Provides an easy way to create a parametric op (OpFactory) with fixed
    (i.e. not dependent on parameters) qubit_count and reg_count."""

    base_id: BaseIdent
    qubit_count: Optional[int]
    reg_count: Optional[int] = 0
    unitary: bool = True
    self_inverse: bool = False
    qregs: Sequence[QRegSpec] | None = None

    def __call__(self, *params: Params.args) -> Op:
        ident = Ident(*self.base_id, params=params)
        qubit_count = self.qubit_count_fn(*params)
        reg_count = self.reg_count_fn(*params)
        unitary = self.unitary
        self_inverse = self.self_inverse_fn(*params)
        qregs = self.qregs_fn(*params)
        return Op(
            id=ident,
            qubit_count=qubit_count,
            reg_count=reg_count,
            unitary=unitary,
            self_inverse=self_inverse,
            qregs=qregs,
        )
