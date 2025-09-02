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
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Optional, Protocol, TypeAlias, cast

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


class SubRepository:
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


_DEFAULT = SubRepository()


def default_repository() -> SubRepository:
    return _DEFAULT


def resolve_sub(op: Op, repository: SubRepository = default_repository()) -> Sub | None:
    resolver = repository.find_resolver(op)
    if resolver:
        return resolver(op, repository)
    else:
        return None


@dataclass
class SubCollector:
    _repository: SubRepository

    def resolve_sub(self, op: Op) -> Sub | None:
        return resolve_sub(op, self._repository)

    def collect_subs(self, op: Op | Sub) -> Mapping[Op, Sub]:
        subs: list[Sub] = []
        sub_map: dict[Op, Optional[Sub]] = {}

        def _collect(op: Op | Sub) -> None:
            if isinstance(op, Op):
                if op in sub_map:
                    return
                logger.info("Resolving: %s", op.id)
                sub = self.resolve_sub(op)
            else:
                sub = op

            if sub is not None:
                logger.debug("Resolved: %s", sub)
                if isinstance(op, Op):
                    sub_map[op] = sub
                if sub not in subs:
                    subs.append(sub)
                    for o, _, _ in sub.operations:
                        _collect(o)
            elif isinstance(op, Op):
                logger.debug("Not found: %s", op.id)
                sub_map[op] = None

        _collect(op)
        return {op: sub for op, sub in sub_map.items() if sub is not None}
