# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar, Union, cast

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401


_T = TypeVar("_T", bound=Union[int, float, complex])


def count_mean(value_counts: Mapping[_T, Union[int, float]]) -> _T:
    """Calculate a mean value from count statistics (mapping from values to
    counts)."""
    values, counts = zip(*value_counts.items())
    sum = np.sum(counts)
    return cast(_T, np.inner(values, counts) / sum)


def count_var(value_counts: Mapping[_T, Union[int, float]]) -> float:
    """Calculate a sample variance from count statistics (mapping from values
    to counts)."""
    mean = count_mean(value_counts)
    values, counts = zip(*value_counts.items())
    sum = np.sum(counts)
    square_errors = np.abs(np.asarray(values) - mean) ** 2
    return cast(float, np.inner(square_errors, counts) / sum)
