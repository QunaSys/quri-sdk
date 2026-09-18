# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deprecated.

Use :mod:`quri_algo.circuit_lib.time_evolution.trotter_time_evo`.
"""

import warnings

warnings.warn(
    "quri_algo.qsub is deprecated. Use quri_algo.circuit_lib instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quri_algo.circuit_lib.time_evolution.trotter_time_evo import (  # noqa: F401, E402
    TrotterTimeEvo,
    TrotterTimeEvoSub,
)

__all__ = ["TrotterTimeEvo", "TrotterTimeEvoSub"]
