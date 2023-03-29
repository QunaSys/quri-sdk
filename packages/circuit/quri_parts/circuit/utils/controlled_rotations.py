# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from .. import (
    LinearMappedUnboundParametricQuantumCircuit,
    Parameter,
    ParameterOrLinearFunction,
)


def add_controlled_RX_gate(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    control_index: int,
    target_index: int,
    param_fn: ParameterOrLinearFunction,
) -> LinearMappedUnboundParametricQuantumCircuit:
    """Add a controlled-RX gate to the given ``circuit``."""
    if isinstance(param_fn, Parameter):
        p_fn = {param_fn: 0.5}
        inv_sign_p_fn = {param_fn: -0.5}
    else:
        p_fn = {param: 0.5 * val for param, val in param_fn.items()}
        inv_sign_p_fn = {param: -0.5 * val for param, val in param_fn.items()}
    circuit.add_RZ_gate(target_index, 0.5 * np.pi)
    circuit.add_ParametricRY_gate(target_index, p_fn)
    circuit.add_CNOT_gate(control_index, target_index)
    circuit.add_ParametricRY_gate(target_index, inv_sign_p_fn)
    circuit.add_CNOT_gate(control_index, target_index)
    circuit.add_RZ_gate(target_index, 0.5 * -np.pi)

    return circuit


def add_controlled_RY_gate(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
    control_index: int,
    target_index: int,
    param_fn: ParameterOrLinearFunction,
) -> LinearMappedUnboundParametricQuantumCircuit:
    """Add a controlled-RY gate to the given ``circuit``."""
    if isinstance(param_fn, Parameter):
        p_fn = {param_fn: 0.5}
        inv_sign_p_fn = {param_fn: -0.5}
    else:
        p_fn = {param: 0.5 * val for param, val in param_fn.items()}
        inv_sign_p_fn = {param: -0.5 * val for param, val in param_fn.items()}
    circuit.add_ParametricRY_gate(target_index, p_fn)
    circuit.add_CNOT_gate(control_index, target_index)
    circuit.add_ParametricRY_gate(target_index, inv_sign_p_fn)
    circuit.add_CNOT_gate(control_index, target_index)

    return circuit
