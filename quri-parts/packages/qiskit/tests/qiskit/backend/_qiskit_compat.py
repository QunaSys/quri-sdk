# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility shims for tests that need to run on both qiskit 1.x and qiskit
2.x.

Several APIs were removed in qiskit 2.0 (and in the matching qiskit-ibm-
runtime releases). This module imports the qiskit-1.x names when
available and falls back to the qiskit-2.x equivalents (or ``None`` for
names with no analog) so the test modules can be collected on both
versions.
"""

from typing import Any

# ---------------------------------------------------------------------------
# BackendV1: removed in qiskit 2.0. ``None`` on qiskit 2.x so V1-only test
# cases can be skipped with ``@pytest.mark.skipif(BackendV1 is None, ...)``.
# ---------------------------------------------------------------------------
try:
    from qiskit.providers.backend import BackendV1
except ImportError:  # qiskit >= 2.0
    BackendV1 = None

# ---------------------------------------------------------------------------
# QasmBackendConfiguration: the ``qiskit.providers.models`` module was removed
# in qiskit 2.0. The tests only use it as a ``Mock`` spec, so on qiskit 2.x we
# fall back to ``object`` (``Mock(spec=object)`` behaves the same way for the
# attributes the tests set explicitly).
# ---------------------------------------------------------------------------
try:
    from qiskit.providers.models import QasmBackendConfiguration
except (ImportError, ModuleNotFoundError):  # qiskit >= 2.0
    QasmBackendConfiguration = object

# ---------------------------------------------------------------------------
# RuntimeJob: replaced by RuntimeJobV2 in qiskit-ibm-runtime; the
# ``qiskit_ibm_runtime.runtime_job`` module was removed alongside it. The
# quri-parts source already uses ``RuntimeJobV2``, so prefer it on both
# versions for consistency, falling back to the legacy class if needed.
# ---------------------------------------------------------------------------
try:
    from qiskit_ibm_runtime import RuntimeJobV2 as RuntimeJob
except ImportError:  # pragma: no cover - very old runtime
    from qiskit_ibm_runtime.runtime_job import RuntimeJob


class _JobStatusFallback:
    """String-valued ``JobStatus`` substitute for qiskit-ibm-runtime >= 0.30.

    On qiskit-ibm-runtime < 0.30 ``JobStatus`` is an ``enum.Enum`` whose
    members compare equal to themselves. ``RuntimeJobV2.status()``
    returns a plain string (e.g. ``"DONE"``) and the quri-parts source
    compares against the string ``"DONE"``. This shim exposes the same
    member names as the old enum but with the string value, so test
    assertions of the form ``status() == JobStatus.DONE`` keep working.
    """

    INITIALIZING = "INITIALIZING"
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    RUNNING = "RUNNING"
    CANCELLED = "CANCELLED"
    DONE = "DONE"
    ERROR = "ERROR"


try:
    # qiskit-ibm-runtime < 0.30 (paired with qiskit 1.x).
    from qiskit_ibm_runtime.runtime_job import JobStatus
except (ImportError, ModuleNotFoundError):  # qiskit-ibm-runtime >= 0.30
    JobStatus = _JobStatusFallback


# ---------------------------------------------------------------------------
# Session(service=...): qiskit-ibm-runtime >= 0.30 dropped the ``service``
# keyword from ``Session.__init__`` (the service is now derived from the
# backend). Tests that explicitly pass ``service`` exercise behavior that has
# no analog on the new runtime, so they are skipped there.
# ---------------------------------------------------------------------------
def _session_accepts_service() -> bool:
    import inspect

    from qiskit_ibm_runtime import Session

    try:
        return "service" in inspect.signature(Session.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


SESSION_ACCEPTS_SERVICE = _session_accepts_service()


def make_mock_config(spec: Any = None, **attrs: Any) -> Any:
    """Build a ``Mock`` backend-configuration object.

    Uses ``QasmBackendConfiguration`` as the spec when it is available
    (qiskit 1.x) and a plain unspecced ``Mock`` otherwise (qiskit 2.x),
    then sets the provided attributes.
    """
    from unittest.mock import Mock

    spec = QasmBackendConfiguration if spec is None else spec
    conf = Mock(spec=spec) if spec is not object else Mock()
    for key, value in attrs.items():
        setattr(conf, key, value)
    return conf


__all__ = [
    "BackendV1",
    "QasmBackendConfiguration",
    "RuntimeJob",
    "JobStatus",
    "SESSION_ACCEPTS_SERVICE",
    "make_mock_config",
]
