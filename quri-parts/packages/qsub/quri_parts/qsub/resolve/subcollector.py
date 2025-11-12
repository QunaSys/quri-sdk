# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Mapping
import logging 
from dataclasses import dataclass
from quri_parts.qsub.op import Op
from quri_parts.qsub.sub import Sub
from .resolve import SubRepository, resolve_sub

logger = logging.getLogger(__name__)


@dataclass
class SubCollector:
    _repository: SubRepository

    def resolve_sub(self, op: Op) -> Sub | None:
        return resolve_sub(op, self._repository)

    def collect_subs(self, op: Op | Sub) -> Mapping[Op, Sub]:
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

                not_computed = set(o for o, _, _ in sub.operations) - set(
                    sub_map.keys()
                )
                for o in not_computed:
                    _collect(o)
            elif isinstance(op, Op):
                logger.debug("Not found: %s", op.id)
                sub_map[op] = None

        _collect(op)
        return {op: sub for op, sub in sub_map.items() if sub is not None}
