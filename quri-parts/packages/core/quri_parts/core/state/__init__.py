# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interfaces and classes representing quantum states."""
from typing import Sequence, TypeVar

from .comp_basis import ComputationalBasisState, comp_basis_superposition
from .state import CircuitQuantumState, GeneralCircuitQuantumState, QuantumState
from .state_helper import apply_circuit, quantum_state
from .state_parametric import ParametricCircuitQuantumState
from .state_vector import QuantumStateVector, StateVectorType
from .state_vector_parametric import ParametricQuantumStateVector

#: A type alias representing a numerical state vector,
#: equivalent to np.ndarray of complex floats.
StateVectorType = StateVectorType

#: A type variable representing *either one of* non-parametric quantum state classes.
#: You can use it as a type parameter for generic classes or functions which can be used
#: with any non-parametric quantum state (i.e. not depending on whether the state is
#: a state vector or not).
QuantumStateT = TypeVar("QuantumStateT", CircuitQuantumState, QuantumStateVector)


#: A type variable representing *either one of* parametric quantum state classes.
#: You can use it as a type parameter for generic classes or functions which can be used
#: with any parametric quantum state (i.e. not depending on whether the state is
#: a state vector or not).
ParametricQuantumStateT = TypeVar(
    "ParametricQuantumStateT",
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
)

#: ComputationalBasisSuperposition represents a state that is formed as a linear
#: combination of :class:`quri_parts.core.state.ComputationalBasisState`\ s.
#: Note that the state expressed in this form is not necessarily normalized.
ComputationalBasisSuperposition = tuple[
    Sequence[complex], Sequence["ComputationalBasisState"]
]


__all__ = [
    "QuantumState",
    "CircuitQuantumState",
    "GeneralCircuitQuantumState",
    "ParametricCircuitQuantumState",
    "ComputationalBasisState",
    "comp_basis_superposition",
    "ComputationalBasisSuperposition",
    "StateVectorType",
    "QuantumStateVector",
    "ParametricQuantumStateVector",
    "QuantumStateT",
    "ParametricQuantumStateT",
    "quantum_state",
    "apply_circuit",
]
