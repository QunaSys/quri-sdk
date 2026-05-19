# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeAlias, cast

from typing_extensions import Self

from quri_parts.qsub.op import BaseIdent, Ident, Op, OpFactory, Params
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.register import Register
from quri_parts.qsub.sub import Sub, SubFactory

logger = logging.getLogger(__name__)


class SubResolver(Protocol):
    def __call__(self, op: Op, repository: "SubRepository") -> Sub | None:
        ...


@dataclass
class SimpleSubResolver(SubResolver, Generic[Params]):
    sub: Sub | SubFactory[Params]

    def __call__(self, op: Op, repository: "SubRepository") -> Sub:
        if isinstance(self.sub, Sub):
            return self.sub
        else:
            return self.sub(*cast(Params.args, op.id.params))


def _get_base_id(op: Op | OpFactory[Params] | BaseIdent) -> BaseIdent:
    if isinstance(op, tuple):
        return op
    else:
        return op.base_id


SubResolverCondition: TypeAlias = Callable[[Ident], bool]


class SubRepository(Protocol):
    @abstractmethod
    def find_resolver(self, op: Op) -> SubResolver | None:
        ...

    @abstractmethod
    def copy(self) -> Self:
        ...

    @abstractmethod
    def with_override(self, addition: "SubRepository") -> "SubRepository":
        ...


class SimpleSubRepository(SubRepository):
    def __init__(self) -> None:
        self._mapping: dict[
            BaseIdent, list[tuple[SubResolver, SubResolverCondition | None]]
        ] = defaultdict(list)

    def find_resolver(self, op: Op) -> SubResolver | None:
        for resolver, cond in reversed(self._mapping[op.base_id]):
            if cond is None:
                return resolver
            if cond(op.id):
                return resolver
        return None

    def register_sub(
        self, op: Op | OpFactory[Any] | BaseIdent, sub: Sub | SubFactory[Any]
    ) -> None:
        resolver = SimpleSubResolver(sub)
        self._mapping[_get_base_id(op)].append((resolver, None))

    def register_sub_resolver(
        self,
        op: Op | OpFactory[Any] | BaseIdent,
        resolver: SubResolver,
        condition: SubResolverCondition | None = None,
    ) -> None:
        self._mapping[_get_base_id(op)].append((resolver, condition))

    def copy(self) -> "SimpleSubRepository":
        ret = SimpleSubRepository()
        for k, v in self._mapping.items():
            ret._mapping[k] = [item for item in v]
        return ret

    def with_override(self, addition: "SubRepository") -> "SubRepository":
        """Creates a new SubRepository that overrides the."""
        return CompositeSubRepository(self, addition)


_DEFAULT = SimpleSubRepository()


def default_repository() -> SimpleSubRepository:
    return _DEFAULT


class CompositeSubRepository(SubRepository):
    """A :class:`SubRepositoryProtocol` that holds the base repo and an
    additional `SubRepository`."""

    def __init__(self, base_repo: SubRepository, addition_repo: SubRepository):
        self._base_repo = base_repo
        self._addition_repo = addition_repo

    @property
    def base_repo(self) -> SubRepository:
        return self._base_repo

    @property
    def addition_repo(self) -> SubRepository:
        return self._addition_repo

    def find_resolver(self, op: Op) -> SubResolver | None:
        """Finds the resolver starting from the addition repo.

        If none exists in the addition, it finds from the base repo.
        """
        resolver = self._addition_repo.find_resolver(op)
        return resolver if resolver else self._base_repo.find_resolver(op)

    def copy(self) -> "CompositeSubRepository":
        return CompositeSubRepository(
            self._base_repo.copy(), self._addition_repo.copy()
        )

    def with_override(self, addition: SubRepository) -> SubRepository:
        return CompositeSubRepository(self, addition)


def resolve_sub(op: Op, repository: SubRepository = default_repository()) -> Sub | None:
    resolver = repository.find_resolver(op)
    if resolver:
        result = resolver(op, repository)
        if result is not None:
            expected_qubits = set(Qubit(i) for i in range(op.qubit_count))
            actual_qubits = set(result.qubits)
            assert (
                actual_qubits == expected_qubits
            ), f"While resolving op {op!r}: expected {op.qubit_count} qubits, but got {len(actual_qubits)}."
            expected_registers = set(Register(i) for i in range(op.reg_count))
            actual_registers = set(result.registers)
            assert (
                actual_registers == expected_registers
            ), f"While resolving op {op!r}: expected {op.reg_count} registers, but got {len(actual_registers)}."
        return result
    else:
        return None
