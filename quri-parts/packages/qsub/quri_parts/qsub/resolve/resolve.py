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
from typing import Any, Generic, Protocol, Sequence, TypeAlias, cast

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


class SubRepositoryProtocol(Protocol):
    @abstractmethod
    def find_resolver(self, op: Op) -> SubResolver | None:
        ...

    @abstractmethod
    def register_sub(
        self, op: Op | OpFactory[Any] | BaseIdent, sub: Sub | SubFactory[Any]
    ) -> None:
        ...

    @abstractmethod
    def register_sub_resolver(
        self,
        op: Op | OpFactory[Any] | BaseIdent,
        resolver: SubResolver,
        condition: SubResolverCondition | None = None,
    ) -> None:
        ...

    @abstractmethod
    def copy(self) -> Self:
        ...

    @abstractmethod
    def chain(self, repos: Sequence["SubRepository"]) -> "CompositeSubRepository":
        ...


class SubRepository(SubRepositoryProtocol):
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

    def copy(self) -> "SubRepository":
        ret = SubRepository()
        for k, v in self._mapping.items():
            ret._mapping[k] = [item for item in v]
        return ret

    def chain(self, repos: Sequence["SubRepository"]) -> "CompositeSubRepository":
        """Concatenate a sequence of addition repos to the this repo and make a
        :class:`CompositeSubRepository`.
        """
        return CompositeSubRepository(additions=repos, root_repo=self)


_DEFAULT = SubRepository()


def default_repository() -> SubRepository:
    return _DEFAULT


class CompositeSubRepository(SubRepositoryProtocol):
    """A :class:`SubRepositoryProtocol` that holds the root repo and a
    sequence of additional `SubRepository`s.
    """

    def __init__(
        self,
        additions: Sequence[SubRepository],
        root_repo: SubRepository = default_repository(),
    ):
        self.root_repo = root_repo
        self._additions = additions
        self._scoped_repo: SubRepository | None = None
        self._is_live = False

    def find_resolver(self, op: Op) -> SubResolver | None:
        """Finds the resolver starting from the last repo in the addition.
        If none exists in the addition, it finds from the root repo.
        """
        for addition in reversed(self._additions):
            resolver = addition.find_resolver(op)
            if resolver is not None:
                return resolver
        return self.root_repo.find_resolver(op)

    def register_sub(
        self, op: Op | OpFactory[Any] | BaseIdent, sub: Sub | SubFactory[Any]
    ) -> None:
        """Registration not allowed for composite sub repository."""
        raise ValueError("Registration is not allowed for CompositeSubRepository.")

    def register_sub_resolver(
        self,
        op: Op | OpFactory[Any] | BaseIdent,
        resolver: SubResolver,
        condition: SubResolverCondition | None = None,
    ) -> None:
        """Registration not allowed for composite sub repository."""
        raise ValueError("Registration is not allowed for CompositeSubRepository.")

    def copy(self) -> "CompositeSubRepository":
        return CompositeSubRepository(
            [a.copy() for a in self._additions], self.root_repo.copy()
        )

    def chain(self, repos: Sequence[SubRepository]) -> "CompositeSubRepository":
        return CompositeSubRepository(
            additions=[*self._additions, *repos], root_repo=self.root_repo
        )


def resolve_sub(op: Op, repository: SubRepository = default_repository()) -> Sub | None:
    resolver = repository.find_resolver(op)
    if resolver:
        return resolver(op, repository)
    else:
        return None
