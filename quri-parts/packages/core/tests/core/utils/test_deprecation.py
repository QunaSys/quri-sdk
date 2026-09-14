# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from quri_parts.core.utils.deprecation import (
    deprecated_measurement_cnt_kwarg,
    deprecated_shots_kwarg,
)


@deprecated_shots_kwarg
def _f(circuit: str, shots: int) -> tuple[str, int]:
    return circuit, shots


@deprecated_measurement_cnt_kwarg
def _g(circuit: str, shots: int) -> tuple[str, int]:
    return circuit, shots


def test_new_kwarg_no_warning(recwarn: pytest.WarningsRecorder) -> None:
    assert _f("c", shots=10) == ("c", 10)
    assert len(recwarn) == 0


def test_old_kwarg_is_mapped_and_warns() -> None:
    with pytest.deprecated_call():
        assert _f("c", n_shots=10) == ("c", 10)  # type: ignore[call-arg]


def test_old_measurement_cnt_kwarg_is_mapped_and_warns() -> None:
    with pytest.deprecated_call():
        assert _g("c", measurement_cnt=10) == ("c", 10)  # type: ignore[call-arg]
