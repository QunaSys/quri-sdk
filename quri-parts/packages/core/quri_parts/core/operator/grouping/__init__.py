# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from typing import Callable, Union

from typing_extensions import TypeAlias

from ..operator import Operator
from ..pauli import CommutablePauliSet, PauliLabel
from .pauli_grouping import (
    bitwise_pauli_grouping,
    individual_pauli_grouping,
    sorted_injection_grouping,
)

#: PauliGrouping represents a function that converts either an :class:`~Operator` or
#: an Iterable of :class:`~PauliLabel` into groups of commutable Pauli operators
#: (``Iterable[CommutablePauliSet]``).
PauliGrouping: TypeAlias = Callable[
    [Union[Operator, Iterable[PauliLabel]]], Iterable[CommutablePauliSet]
]

__all__ = [
    "PauliGrouping",
    "individual_pauli_grouping",
    "bitwise_pauli_grouping",
    "sorted_injection_grouping",
]
