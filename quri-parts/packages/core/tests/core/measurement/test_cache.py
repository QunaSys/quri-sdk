# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock

from quri_parts.core.measurement import (
    CachedMeasurementFactory,
    bitwise_commuting_pauli_measurement,
)
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label


def test_cached_measurement_factory() -> None:
    operator_1 = Operator(
        {
            pauli_label("X0 Y1"): 1,
            pauli_label("X0 Z2"): 2,
            pauli_label("Y0 Z2"): 3,
            PAULI_IDENTITY: 4,
        }
    )
    expected_group_1 = bitwise_commuting_pauli_measurement(operator_1)

    operator_2 = Operator(
        {
            pauli_label("X0 Y1"): 1,
            pauli_label("X0 Z2"): 1,
            pauli_label("Y0 Z2"): 1,
            PAULI_IDENTITY: 1,
        }
    )
    expected_group_2 = bitwise_commuting_pauli_measurement(operator_2)

    paulis = [
        pauli_label("X0 Y1"),
        pauli_label("X0 Z2"),
        pauli_label("Y0 Z2"),
        PAULI_IDENTITY,
    ]
    expected_group_3 = bitwise_commuting_pauli_measurement(paulis)

    measurement_factory = Mock(side_effect=bitwise_commuting_pauli_measurement)
    cached_measurement_factory = CachedMeasurementFactory(measurement_factory)
    assert len(cached_measurement_factory.cached_groups) == 0
    assert measurement_factory.call_count == 0

    group_1 = cached_measurement_factory(operator_1)
    assert group_1 == expected_group_1
    assert len(cached_measurement_factory.cached_groups) == 1
    assert measurement_factory.call_count == 1

    group_1_second_run = cached_measurement_factory(operator_1)
    assert group_1_second_run == expected_group_1
    assert len(cached_measurement_factory.cached_groups) == 1
    assert measurement_factory.call_count == 1

    group_2 = cached_measurement_factory(operator_2)
    assert group_2 == expected_group_2
    assert len(cached_measurement_factory.cached_groups) == 2
    assert measurement_factory.call_count == 2

    group_2_second_run = cached_measurement_factory(operator_2)
    assert group_2_second_run == expected_group_2
    assert len(cached_measurement_factory.cached_groups) == 2
    assert measurement_factory.call_count == 2

    group_2_third_run = cached_measurement_factory(operator_2)
    assert group_2_third_run == expected_group_2
    assert len(cached_measurement_factory.cached_groups) == 2
    assert measurement_factory.call_count == 2

    group_3 = cached_measurement_factory(paulis)
    assert group_3 == expected_group_3
    assert len(cached_measurement_factory.cached_groups) == 2
    assert measurement_factory.call_count == 2

    group_3_second_run = cached_measurement_factory(paulis)
    assert group_3_second_run == expected_group_3
    assert len(cached_measurement_factory.cached_groups) == 2
    assert measurement_factory.call_count == 2
