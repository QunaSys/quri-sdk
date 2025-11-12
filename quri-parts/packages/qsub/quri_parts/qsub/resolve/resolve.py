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
import functools
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Generic, Optional, Protocol, Sequence, Type, TypeAlias, cast
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

    @property
    @abstractmethod
    def additions(self) -> Sequence["SubRepository"] | None:
        ...

    @abstractmethod
    def chain(self, repos: Sequence["SubRepository"]) -> "CompositeSubRepository":
        ...

    @abstractmethod
    def flatten(self) -> "SubRepository":
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

    @property
    def additions(self) -> Sequence["SubRepository"] | None:
        return None

    def chain(self, repos: Sequence["SubRepository"]) -> "CompositeSubRepository":
        return CompositeSubRepository(additions=repos, root_repo=self)

    def flatten(self) -> "SubRepository":
        return self.copy()

    def __add__(self, repo: "SubRepository") -> "SubRepository":
        new_repo = self.copy()
        for base_id, res_list in repo._mapping.items():
            new_repo._mapping[base_id].extend(res_list)
        return new_repo


_DEFAULT = SubRepository()


def default_repository() -> SubRepository:
    return _DEFAULT


class CompositeSubRepository(SubRepositoryProtocol):
    def __init__(
        self,
        additions: Sequence[SubRepository],
        root_repo: SubRepository = default_repository(),
    ):
        self.root_repo = root_repo
        self._additions = additions
        self._scoped_repo: SubRepository | None = None
        self._is_live = False

    def __enter__(self) -> "CompositeSubRepository":
        logger.debug("__enter__")
        self._is_live = True
        self._scoped_repo = self.flatten()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ):
        logger.debug("__exit__")
        self._is_live = False
        self._scoped_repo = None

    def find_resolver(self, op: Op) -> SubResolver | None:
        if self._is_live:
            assert self._scoped_repo is not None
            repo_to_use = self._scoped_repo
        else:
            repo_to_use = self.flatten()
        return repo_to_use.find_resolver(op)

    def register_sub(
        self, op: Op | OpFactory[Any] | BaseIdent, sub: Sub | SubFactory[Any]
    ) -> None:
        raise ValueError("Registration is not allowed for CompositeSubRepository.")

    def register_sub_resolver(
        self,
        op: Op | OpFactory[Any] | BaseIdent,
        resolver: SubResolver,
        condition: SubResolverCondition | None = None,
    ) -> None:
        raise ValueError("Registration is not allowed for CompositeSubRepository.")

    def copy(self) -> "CompositeSubRepository":
        return CompositeSubRepository(
            [a.copy() for a in self.additions], self.root_repo.copy()
        )

    @property
    def additions(self) -> Sequence["SubRepository"] | None:
        return self._additions

    def flatten(self) -> "SubRepository":
        logger.debug("Building composed repo.")
        assert self.additions is not None
        return functools.reduce(lambda a, b: a + b, [self.root_repo, *self.additions])

    def chain(self, repos: Sequence[SubRepository]) -> "CompositeSubRepository":
        return CompositeSubRepository(additions=self._additions + repos, root_repo=self)


def resolve_sub(op: Op, repository: SubRepository = default_repository()) -> Sub | None:
    resolver = repository.find_resolver(op)
    if resolver:
        return resolver(op, repository)
    else:
        return None
