# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Circuit and gate inversion (:mod:`quri_parts.circuit.inverse`)
==================================================================

Utilities for computing the inverse (adjoint) of individual gates and
of entire non-parametric circuits.

Key functions (for quick orientation):

- :func:`inverse_gate` -- return the adjoint of a single gate
- :func:`inverse_circuit` -- return a circuit with reversed, inverted gates
"""

from quri_parts.rust.circuit.inverse import inverse_circuit, inverse_gate

__all__ = ["inverse_gate", "inverse_circuit"]
