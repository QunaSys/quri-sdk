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
    def with_override(self, addition: "SimpleSubRepository") -> "SubRepository":
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

    def with_override(
        self, addition: "SimpleSubRepository"
    ) -> "CompositeSubRepository":
        """Creates a new SubRepository that overrides the"""
        return CompositeSubRepository(self, addition)


_DEFAULT = SimpleSubRepository()


def default_repository() -> SimpleSubRepository:
    return _DEFAULT


class CompositeSubRepository(SubRepository):
    """A :class:`SubRepositoryProtocol` that holds the parent repo and an
    additional child `SubRepository`."""

    def __init__(self, parent_repo: SubRepository, child_repo: SimpleSubRepository):
        self.parent_repo = parent_repo
        self.child_repo = child_repo

    def find_resolver(self, op: Op) -> SubResolver | None:
        """Finds the resolver starting from the child repo.

        If none exists in the child, it finds from the parent repo.
        """
        resolver = self.child_repo.find_resolver(op)
        return resolver if resolver else self.parent_repo.find_resolver(op)

    def copy(self) -> "CompositeSubRepository":
        return CompositeSubRepository(self.parent_repo.copy(), self.child_repo.copy())

    def with_override(self, addition: SimpleSubRepository) -> "CompositeSubRepository":
        return CompositeSubRepository(self, addition)


def resolve_sub(op: Op, repository: SubRepository = default_repository()) -> Sub | None:
    resolver = repository.find_resolver(op)
    if resolver:
        return resolver(op, repository)
    else:
        return None
