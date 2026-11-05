# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from contextlib import contextmanager
from typing import Any, ContextManager, Iterator

import pytest

from quri_parts.core.utils.deprecation import (
    deprecated_kwarg,
    deprecated_measurement_cnt_kwarg,
    deprecated_shots_kwarg,
)


@deprecated_shots_kwarg
def sample_with_n_shots_alias(circuit: str, shots: int) -> tuple[str, int]:
    return circuit, shots


@deprecated_measurement_cnt_kwarg
def sample_with_measurement_cnt_alias(circuit: str, shots: int) -> tuple[str, int]:
    return circuit, shots


@deprecated_kwarg("old_name", "new_name")
def identity_with_old_name_alias(new_name: int) -> int:
    return new_name


def test_n_shots_is_mapped_and_warns() -> None:
    with pytest.deprecated_call():
        assert sample_with_n_shots_alias("c", n_shots=10) == (  # type: ignore[call-arg]
            "c",
            10,
        )


def test_measurement_cnt_is_mapped_and_warns() -> None:
    with pytest.deprecated_call():
        assert sample_with_measurement_cnt_alias(  # type: ignore[call-arg]
            "c", measurement_cnt=10
        ) == ("c", 10)


@contextmanager
def no_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.mark.parametrize(
    "args, kwargs, expectation",
    [
        ((), {}, pytest.raises(TypeError)),
        ((), {"new_name": 10}, no_warnings()),
        ((), {"old_name": 10}, pytest.deprecated_call()),
        ((), {"new_name": 10, "old_name": 20}, pytest.raises(TypeError)),
        ((10,), {"old_name": 20}, pytest.raises(TypeError)),
    ],
    ids=["neither", "new-only", "old-only", "both", "positional-and-old"],
)
def test_deprecated_kwarg(
    args: tuple[int, ...], kwargs: dict[str, int], expectation: ContextManager[Any]
) -> None:
    with expectation:
        assert identity_with_old_name_alias(*args, **kwargs) == 10
