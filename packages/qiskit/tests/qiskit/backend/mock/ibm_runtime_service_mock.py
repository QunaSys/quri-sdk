# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# The functionality of `mock_get_backend` contained in the file has been
# modified from the original. The original file is available at:
# https://github.com/Qiskit/qiskit-ibm-runtime/blob/main/qiskit_ibm_runtime/test/ibm_runtime_service_mock.py
"""Mock for qiskit_ibm_runtime.QiskitRuntimeService."""

from unittest.mock import MagicMock

from qiskit.providers.models import QasmBackendConfiguration
from qiskit_ibm_runtime import IBMBackend, QiskitRuntimeService
from qiskit_ibm_runtime.api.clients import RuntimeClient


def mock_get_backend(is_simulator: bool = True) -> QiskitRuntimeService:
    """Mock for QiskitRuntimeService.

    Create a mock of qiskit_ibm_runtime.QiskitRuntimeService that returns
    a single backend. This will not effect the imported qiskit_ibm_runtime.
    QiskitRuntimeService. It is intended to run tests without requiring
    to configure accounts, such as on github CI.

    Args:
        is_simulator: (bool) An argument that determines if the mock backend
            mocks a simulator backend or not.

    Returns: Mock of qiskit_ibm_runtime.QiskitRuntimeService

    Raises:
        NameError: If the specified value of backend
    """
    mock_qiskit_runtime_service = MagicMock(spec=QiskitRuntimeService)
    mock_qiskit_runtime_service._channel_strategy = "fake_strategy"

    conf = MagicMock(spec=QasmBackendConfiguration)
    conf.max_shots = int(1e6)
    conf.simulator = is_simulator

    fake_backend = MagicMock(spec=IBMBackend)
    fake_backend._instance = None
    fake_backend.configuration.return_value = conf
    fake_backend.version = 0
    fake_backend.target.num_qubits = 127

    mock_qiskit_runtime_service.backend.return_value = fake_backend
    mock_qiskit_runtime_service.return_value = mock_qiskit_runtime_service

    client = MagicMock(spec=RuntimeClient)
    mock_qiskit_runtime_service._api_client = client

    return mock_qiskit_runtime_service
