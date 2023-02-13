# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .circuit import (
    GateSequence,
    ImmutableQuantumCircuit,
    MutableQuantumCircuitProtocol,
    NonParametricQuantumCircuit,
    QuantumCircuit,
    QuantumCircuitProtocol,
)
from .circuit_linear_mapped import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuitBase,
)
from .circuit_parametric import (
    CONST,
    ImmutableBoundParametricQuantumCircuit,
    ImmutableUnboundParametricQuantumCircuit,
    MutableUnboundParametricQuantumCircuitProtocol,
    Parameter,
    UnboundParametricQuantumCircuit,
    UnboundParametricQuantumCircuitBase,
    UnboundParametricQuantumCircuitProtocol,
)
from .clifford_gate import is_clifford
from .gate import ParametricQuantumGate, QuantumGate
from .gates import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    TOFFOLI,
    U1,
    U2,
    U3,
    H,
    Identity,
    ParametricPauliRotation,
    ParametricRX,
    ParametricRY,
    ParametricRZ,
    Pauli,
    PauliRotation,
    S,
    Sdag,
    SingleQubitUnitaryMatrix,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    T,
    Tdag,
    TwoQubitUnitaryMatrix,
    UnitaryMatrix,
    X,
    Y,
    Z,
)
from .inverse_gate import inverse_gate
from .parameter_mapping import (
    LinearParameterFunction,
    LinearParameterMapping,
    ParameterMapping,
    ParameterOrLinearFunction,
)

#: A placeholder representing a constant term.
CONST = CONST

#: A type representing a linear (affine) function of parameters. It is an alias for
#: mapping from :class:`~Parameter` to float coefficients. A constant term can be
#: represented as a coefficient for ``CONST``.
LinearParameterFunction = LinearParameterFunction


__all__ = [
    "QuantumGate",
    "ParametricQuantumGate",
    "Identity",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "Sdag",
    "SqrtX",
    "SqrtXdag",
    "SqrtY",
    "SqrtYdag",
    "T",
    "Tdag",
    "RX",
    "RY",
    "RZ",
    "U1",
    "U2",
    "U3",
    "UnitaryMatrix",
    "SingleQubitUnitaryMatrix",
    "TwoQubitUnitaryMatrix",
    "CNOT",
    "CZ",
    "SWAP",
    "TOFFOLI",
    "Pauli",
    "PauliRotation",
    "ParametricRX",
    "ParametricRY",
    "ParametricRZ",
    "ParametricPauliRotation",
    "QuantumCircuitProtocol",
    "MutableQuantumCircuitProtocol",
    "GateSequence",
    "NonParametricQuantumCircuit",
    "QuantumCircuit",
    "ImmutableQuantumCircuit",
    "Parameter",
    "ParameterMapping",
    "ParameterOrLinearFunction",
    "CONST",
    "UnboundParametricQuantumCircuitProtocol",
    "MutableUnboundParametricQuantumCircuitProtocol",
    "UnboundParametricQuantumCircuitBase",
    "UnboundParametricQuantumCircuit",
    "ImmutableUnboundParametricQuantumCircuit",
    "ImmutableBoundParametricQuantumCircuit",
    "LinearParameterFunction",
    "LinearParameterMapping",
    "LinearMappedUnboundParametricQuantumCircuitBase",
    "LinearMappedUnboundParametricQuantumCircuit",
    "ImmutableLinearMappedUnboundParametricQuantumCircuit",
    "inverse_gate",
    "is_clifford",
]
