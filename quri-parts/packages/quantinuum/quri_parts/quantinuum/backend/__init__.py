# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sampling backend for Quantinuum devices accessed through Quantinuum Nexus.

This subpackage requires the optional ``nexus`` dependencies::

    pip install "quri-parts-quantinuum[nexus]"
"""

try:
    import qnexus  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "quri_parts.quantinuum.backend requires the 'qnexus' package. "
        'Install it with: pip install "quri-parts-quantinuum[nexus]"'
    ) from e

from .sampling import (
    QuantinuumSamplingBackend,
    QuantinuumSamplingJob,
    QuantinuumSamplingResult,
    TketCircuitConverter,
    backend_result_to_sampling_counts,
    default_backend_config,
    distribute_shots,
)

__all__ = [
    "QuantinuumSamplingBackend",
    "QuantinuumSamplingJob",
    "QuantinuumSamplingResult",
    "TketCircuitConverter",
    "backend_result_to_sampling_counts",
    "default_backend_config",
    "distribute_shots",
]
