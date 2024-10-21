# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import LinearMappedParametricQuantumCircuit, Parameter
from quri_parts.core.operator import PAULI_IDENTITY, Operator


def add_parametric_commuting_paulis_exp_gate(
    circuit: LinearMappedParametricQuantumCircuit,
    param_fn: dict[Parameter, float],
    qp_operator: Operator,
    coeff: float = 1,
) -> None:
    """Add exponential pauli rotation gate to a
    :class:`~LinearMappedParametricQuantumCircuit` in place
    according to the equation:

    .. math::
        \\exp \\left[
            i \\text{c} * f(\\theta_1, \\theta_2, \\cdots) * \\text{qp_operator}
        \\right]

    Arg:
        circuit:
            The circuit to add a pauli exponential rotation gate to.
        param_fn:
            A dict representing parametric function in front of the qp_operator.
        qp_operator:
            A Hermitian quri-parts operator.
        coeff:
            An overall real coeffcient in the exponent.
    """
    for pauli, op_coeff in qp_operator.items():
        if pauli == PAULI_IDENTITY:
            # This corresponds to a global phase.
            continue
        pauli_index_list, pauli_id_list = zip(*pauli)
        new_param_mapping = {
            param: -2 * op_coeff.real * old_coeff * coeff
            for param, old_coeff in param_fn.items()
        }
        circuit.add_ParametricPauliRotation_gate(
            pauli_index_list, pauli_id_list, new_param_mapping
        )
