# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "sAS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional
from unittest.mock import patch

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import Session

from .ibm_runtime_service_mock import mock_get_backend


class TestMockRuntimeService:
    @patch("qiskit_ibm_runtime.SamplerV2._validate_options", return_value=None)
    def test_fake_service_call(self, _: Optional[Any] = None) -> None:
        # Checking if fake backend works
        runtime_service = mock_get_backend()
        service = runtime_service()
        with Session(service=service, backend=runtime_service.backend()) as session:
            sampler = Sampler(session=session)
            bell = QuantumCircuit(2)
            bell.h(0)
            bell.cx(0, 1)
            bell.measure_all(inplace=True)

            _ = sampler.run([bell], shots=1000)

        service.run.assert_called_once()
        args_list = service.run.call_args_list
        assert len(args_list) == 1
        _, kwargs = args_list[0]
        shots = kwargs["inputs"]["pubs"][0].shots
        assert shots == 1000
