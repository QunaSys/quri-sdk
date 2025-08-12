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

import pytest
from qiskit.providers.backend import Backend, BackendV1, BackendV2
from qiskit.providers.models import QasmBackendConfiguration
from qiskit_ibm_runtime import IBMBackend

from quri_parts.backend import BackendError
from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit
from quri_parts.qiskit.backend import (
    QiskitSavedDataSamplingJob,
    QiskitSavedDataSamplingResult,
    convert_qiskit_sampling_count_to_qp_sampling_count,
    distribute_backend_shots,
    get_backend_min_max_shot,
    get_job_mapper_and_circuit_transpiler,
)
from quri_parts.qiskit.backend.utils import DEFAULT_MAX_SHOT


class TestDistributeBackendShots:
    def test_min_shot_1_disable_shots_roundup(self) -> None:
        shot_distribution_0 = distribute_backend_shots(
            1080, 1, 100, enable_shots_roundup=False
        )
        assert shot_distribution_0 == [100] * 10 + [80]

    def test_min_shot_1_enable_shots_roundup(self) -> None:
        shot_distribution = distribute_backend_shots(
            1080, 1, 100, enable_shots_roundup=True
        )
        assert shot_distribution == [100] * 10 + [80]

    def test_min_shot_30_disable_shots_roundup(self) -> None:
        shot_distribution = distribute_backend_shots(
            1020, 30, 100, enable_shots_roundup=False
        )
        assert shot_distribution == [100] * 10

    def test_min_shot_30_enable_shots_roundup(self) -> None:
        shot_distribution = distribute_backend_shots(
            1020, 30, 100, enable_shots_roundup=True
        )
        assert shot_distribution == [100] * 10 + [30]

    def test_shot_split_evenly_disable_shots_roundup(self) -> None:
        shot_distribution = distribute_backend_shots(
            200, 1, 100, enable_shots_roundup=False
        )
        assert shot_distribution == [100, 100]

    def test_shot_split_evenly_enable_shots_roundup(self) -> None:
        shot_distribution = distribute_backend_shots(
            200, 1, 100, enable_shots_roundup=True
        )
        assert shot_distribution == [100, 100]


class TestGetJobMapperAndCircuitTranspiler:
    raw_data = {"00": 1, "01": 2, "10": 3, "11": 4}
    saved_result = QiskitSavedDataSamplingResult(raw_data=raw_data)
    saved_job = QiskitSavedDataSamplingJob(
        circuit_qasm="test_circuit_str", n_shots=10, saved_result=saved_result
    )
    circuit = QuantumCircuit(2)

    @staticmethod
    def circuit_transpiler(
        _: NonParametricQuantumCircuit,
    ) -> NonParametricQuantumCircuit:
        circuit = QuantumCircuit(2)
        circuit.add_X_gate(0)
        circuit.add_X_gate(0)
        circuit.add_Y_gate(1)
        circuit.add_Y_gate(1)
        return circuit

    def test_no_mapper_and_no_transpiler(self) -> None:
        # qubit_mapping is None, circuit_transpiler is None
        job_mapper, circuit_transpiler = get_job_mapper_and_circuit_transpiler()
        assert job_mapper(self.saved_job).result().counts == {0: 1, 1: 2, 2: 3, 3: 4}
        assert circuit_transpiler(self.circuit) == self.circuit

    def test_non_trivial_mapper_and_no_transpiler(self) -> None:
        # qubit_mapping is not None, circuit_transpiler is None
        job_mapper, circuit_transpiler = get_job_mapper_and_circuit_transpiler(
            qubit_mapping={0: 1, 1: 0}
        )
        assert job_mapper(self.saved_job).result().counts == {0: 1, 2: 2, 1: 3, 3: 4}
        assert circuit_transpiler(self.circuit) == self.circuit

    def test_no_mapper_and_non_trivial_transpiler(self) -> None:
        # qubit_mapping is None, circuit_transpiler is not None
        job_mapper, circuit_transpiler = get_job_mapper_and_circuit_transpiler(
            circuit_transpiler=self.circuit_transpiler
        )
        assert job_mapper(self.saved_job).result().counts == {0: 1, 1: 2, 2: 3, 3: 4}
        expected_transpiled_circuit = QuantumCircuit(2)
        expected_transpiled_circuit.add_X_gate(0)
        expected_transpiled_circuit.add_X_gate(0)
        expected_transpiled_circuit.add_Y_gate(1)
        expected_transpiled_circuit.add_Y_gate(1)
        assert circuit_transpiler(self.circuit) == expected_transpiled_circuit

    def test_non_trivial_mapper_and_non_trivial_transpiler(self) -> None:
        # qubit_mapping is not None, circuit_transpiler is not None
        job_mapper, circuit_transpiler = get_job_mapper_and_circuit_transpiler(
            qubit_mapping={0: 1, 1: 0}, circuit_transpiler=self.circuit_transpiler
        )
        assert job_mapper(self.saved_job).result().counts == {0: 1, 2: 2, 1: 3, 3: 4}
        expected_transpiled_circuit = QuantumCircuit(2)
        expected_transpiled_circuit.add_X_gate(1)
        expected_transpiled_circuit.add_X_gate(1)
        expected_transpiled_circuit.add_Y_gate(0)
        expected_transpiled_circuit.add_Y_gate(0)
        assert circuit_transpiler(self.circuit) == expected_transpiled_circuit


def test_convert_qiskit_sampling_count_to_qp_sampling_count() -> None:
    converted_sampling_count = convert_qiskit_sampling_count_to_qp_sampling_count(
        {"00": 1, "01": 2, "10": 3, "11": 4}
    )
    assert converted_sampling_count == {0: 1, 1: 2, 2: 3, 3: 4}


def test_get_backend_min_max_shot() -> None:
    backend = Mock(spec=Backend)
    with pytest.raises(BackendError, match="Backend not supported"):
        get_backend_min_max_shot(backend)

    backend = Mock(spec=BackendV1)
    conf = Mock(spec=QasmBackendConfiguration)
    conf.max_shots = int(1e3)
    backend.configuration.return_value = conf
    min_shots, max_shots = get_backend_min_max_shot(backend)
    assert min_shots == 1
    assert max_shots == int(1e3)

    backend = Mock(spec=BackendV1)
    conf = Mock(spec=QasmBackendConfiguration)
    backend.configuration.return_value = conf
    with pytest.warns(
        UserWarning,
        match=(
            "No max_shots setting is found. "
            "The max shot is set to default value 1000000"
        ),
    ):
        min_shots, max_shots = get_backend_min_max_shot(backend)
        assert min_shots == 1
        assert max_shots == DEFAULT_MAX_SHOT

    backend = Mock(spec=BackendV1)
    conf = Mock(spec=QasmBackendConfiguration)
    conf.max_shots = 0
    backend.configuration.return_value = conf
    with pytest.warns(
        UserWarning,
        match=(
            "No max_shots setting is found. "
            "The max shot is set to default value 1000000"
        ),
    ):
        min_shots, max_shots = get_backend_min_max_shot(backend)
        assert min_shots == 1
        assert max_shots == DEFAULT_MAX_SHOT

    backend = Mock(spec=BackendV2)
    backend.max_shots = 10
    min_shots, max_shots = get_backend_min_max_shot(backend)
    assert min_shots == 1
    assert max_shots == 10

    backend = Mock(spec=BackendV2)
    with pytest.warns(
        UserWarning,
        match=(
            "No max_shots setting is found. "
            "The max shot is set to default value 1000000"
        ),
    ):
        min_shots, max_shots = get_backend_min_max_shot(backend)
        assert min_shots == 1
        assert max_shots == DEFAULT_MAX_SHOT

    backend = Mock(spec=BackendV2)
    backend.max_shots = 0
    with pytest.warns(
        UserWarning,
        match=(
            "No max_shots setting is found. "
            "The max shot is set to default value 1000000"
        ),
    ):
        min_shots, max_shots = get_backend_min_max_shot(backend)
        assert min_shots == 1
        assert max_shots == DEFAULT_MAX_SHOT

    backend = Mock(spec=IBMBackend)
    conf = Mock(spec=QasmBackendConfiguration)
    conf.max_shots = int(1e3)
    backend.configuration.return_value = conf
    min_shots, max_shots = get_backend_min_max_shot(backend)
    assert min_shots == 1
    assert max_shots == 1000

    backend = Mock(spec=IBMBackend)
    conf = Mock(spec=QasmBackendConfiguration)
    conf.max_shots = 0
    backend.configuration.return_value = conf
    with pytest.warns(
        UserWarning,
        match=(
            "No max_shots setting is found. "
            "The max shot is set to default value 1000000"
        ),
    ):
        min_shots, max_shots = get_backend_min_max_shot(backend)
    assert min_shots == 1
    assert max_shots == DEFAULT_MAX_SHOT
