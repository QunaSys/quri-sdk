# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.core.estimator import Estimatable
from quri_parts.core.operator import PAULI_IDENTITY, Operator, PauliLabel, zero
from quri_parts.core.state import QuantumState


def is_estimatable(observable: Estimatable, state: QuantumState) -> bool:
    """Check if the qubit count of the observable is larger than that of the
    state."""
    if observable == PAULI_IDENTITY or observable == zero():
        return True

    elif isinstance(observable, PauliLabel):
        min_state_qubit_required = max(observable.qubit_indices()) + 1
        return min_state_qubit_required <= state.qubit_count

    elif isinstance(observable, Operator):
        for op in observable:
            if not is_estimatable(op, state):
                return False
        return True

    else:
        assert False, "Observable should be either a PauliLabel or an Operator."
