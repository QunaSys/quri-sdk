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

from qiskit.test import mock as backend_mocks
from qiskit_ibm_runtime import QiskitRuntimeService


def mock_get_backend(backend: str) -> QiskitRuntimeService:
    """Mock for QiskitRuntimeService.

    Create a mock of qiskit_ibm_runtime.QiskitRuntimeService that returns
    a single backend. This will not effect the imported qiskit_ibm_runtime.
    QiskitRuntimeService. It is intended to run tests without requiring
    to configure accounts, such as on github CI.

    Args:
        backend (str): The class name as a string for the fake device to
            return. For example, FakeVigo.

    Returns: Mock of qiskit_ibm_runtime.QiskitRuntimeService

    Raises:
        NameError: If the specified value of backend
    """
    mock_qiskit_runtime_service = MagicMock()
    if not hasattr(backend_mocks, backend):
        raise NameError(
            "The specified backend is not a valid mock from qiskit.test.mock"
        )
    fake_backend = getattr(backend_mocks, backend)()
    mock_qiskit_runtime_service.backend.return_value = fake_backend
    mock_qiskit_runtime_service.return_value = mock_qiskit_runtime_service
    return mock_qiskit_runtime_service
