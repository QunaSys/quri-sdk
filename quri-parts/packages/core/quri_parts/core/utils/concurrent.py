# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

if TYPE_CHECKING:
    from concurrent.futures import Executor

T_common = TypeVar("T_common")
T_individual = TypeVar("T_individual")
R = TypeVar("R")


def execute_concurrently(
    fn: Callable[[T_common, Sequence[T_individual]], Iterable[R]],
    common_input: T_common,
    individual_inputs: Iterable[T_individual],
    executor: Optional["Executor"] = None,
    concurrency: int = 1,
) -> Sequence[R]:
    """Execute the given function for each input concurrently and returns all
    the results as a sequence.

    If executor is None, the executions are performed in sequence. If an
    executor is given, the function is executed for each input
    concurrently using :meth:`Executor.map`.
    """
    individual_input_list = list(individual_inputs)

    if executor is None:
        return list(fn(common_input, individual_input_list))

    input_counts = [
        (len(individual_input_list) + i) // concurrency for i in range(concurrency)
    ]
    input_list = [
        individual_input_list[
            sum(input_counts[:i]) : sum(input_counts[: i + 1])  # noqa: E203
        ]
        for i in range(concurrency)
    ]
    results = executor.map(fn, [common_input] * concurrency, input_list)

    return list(chain.from_iterable(results))
