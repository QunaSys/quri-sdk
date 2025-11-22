# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Callable, Protocol

from typing_extensions import TypeAlias

from quri_parts.circuit import (
    ImmutableQuantumCircuit,
    LinearMappedParametricQuantumCircuit,
    LinearParameterMapping,
    ParametricQuantumCircuitProtocol,
    QuantumCircuit,
    QuantumGate,
    gate_names,
)

#: CircuitTranspiler Interface. A function or callable object that can map
#: ImmutableQuantumCircuit to ImmutableQuantumCircuit.
CircuitTranspiler: TypeAlias = Callable[
    [ImmutableQuantumCircuit], ImmutableQuantumCircuit
]


class CircuitTranspilerProtocol(Protocol):
    """Protocol of callable class that transpiles ImmutableQuantumCircuit to
    ImmutableQuantumCircuit."""

    @abstractmethod
    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        ...


class SequentialTranspiler(CircuitTranspilerProtocol):
    """CircuitTranspiler, which applies CircuitTranspilers in sequence.

    Args:
        transpilers: Sequence of CircuitTranspilers.

    Examples:
        .. highlight:: python
        .. code-block:: python

            transpiler = SequentialTranspiler(
                [
                    ATranspiler(),
                    BTranspiler(arg..),
                    ..
                ]
            )
            circuit = transpiler(circuit)
    """

    def __init__(self, transpilers: Sequence[CircuitTranspiler]):
        self._transpilers = transpilers

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        for transpiler in self._transpilers:
            circuit = transpiler(circuit)
        return circuit


class GateDecomposer(CircuitTranspilerProtocol, ABC):
    """Abstract class that represents CircuitTranspiler, such that target gates
    are selected by decision function and each target gate is replaced by a
    sequence of multiple gates."""

    @abstractmethod
    def is_target_gate(self, gate: QuantumGate) -> bool:
        """Determine if a given gate is subject to decomposition.

        Args:
            gate: Gates in the circuit that are scanned from the front.
        """
        ...

    @abstractmethod
    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        """Describe the specific decomposition process. Only the target gates
        satisfying is_target_gate() method are passed.

        Args:
            gate: The gates to be decomposed.
        """
        ...

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        cg: list[QuantumGate] = []
        for gate in circuit.gates:
            if self.is_target_gate(gate):
                cg.extend(self.decompose(gate))
            else:
                cg.append(gate)

        cc = QuantumCircuit(circuit.qubit_count, cbit_count=circuit.cbit_count)

        cc.extend(cg)
        return cc


class GateKindDecomposer(GateDecomposer, ABC):
    """Abstract class that represents CircuitTranspiler, such that each gate is
    identified by its gate name and the target gate is replaced by a sequence
    of multiple gates.

    Classes inheriting from this class can be used for
    ParallelDecomposer.
    """

    @property
    @abstractmethod
    def target_gate_names(self) -> Sequence[str]:
        """Returns the set of gate names to be decomposed."""
        ...

    def is_target_gate(self, gate: QuantumGate) -> bool:
        return gate.name in self.target_gate_names


class ParallelDecomposer(CircuitTranspilerProtocol):
    """CircuitTranspiler, which executes given GateKindDecomposer within a
    single loop traversing the gate from the front.

    Args:
        decomposers: Sequence of GateKindDecomposer with no duplicate gate types
            to act on.
    """

    def __init__(self, decomposers: Sequence[GateKindDecomposer]):
        self._decomposer_map: Mapping[str, GateKindDecomposer] = {}
        for dc in decomposers:
            for name in dc.target_gate_names:
                if name in self._decomposer_map:
                    raise ValueError(
                        "Multiple transpilers acting on the same type of gate"
                        " were given."
                    )
                self._decomposer_map[name] = dc

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        cg: list[QuantumGate] = []
        for gate in circuit.gates:
            if gate.name in self._decomposer_map:
                cg.extend(self._decomposer_map[gate.name].decompose(gate))
            else:
                cg.append(gate)

        cc = QuantumCircuit(circuit.qubit_count, cbit_count=circuit.cbit_count)

        cc.extend(cg)
        return cc


#: ParametricCircuitTranspiler Interface. A function or callable object that can map
#: ParametricQuantumCircuit to ParametricQuantumCircuit.
ParametricCircuitTranspiler: TypeAlias = Callable[
    [ParametricQuantumCircuitProtocol],
    ParametricQuantumCircuitProtocol,
]


class ParametricCircuitTranspilerProtocol(Protocol):
    """Protocol of callable class that transpiles ParametricQuantumCircuit to
    ParametricQuantumCircuit."""

    def __call__(
        self, circuit: ParametricQuantumCircuitProtocol
    ) -> ParametricQuantumCircuitProtocol:
        ...


class ParametricTranspiler(ParametricCircuitTranspilerProtocol):
    """Circuit transpiler for ParametricQuantumCircuitProtocol, which wraps
    CircuitTranspiler and allows it to be applied to
    ParametricQuantumCircuitProtocol.

    Each successive QuantumGate sequence in the circuit is transpiled respectively and
    the results are concatenated. ParametricQuantumGates and parameter mappings are
    replicated intact.

    Args:
        transpiler: CircuitTranspiler to be wrapped.
    """

    def __init__(self, transpiler: CircuitTranspiler):
        self._transpiler = transpiler

    def __call__(
        self, circuit: ParametricQuantumCircuitProtocol
    ) -> LinearMappedParametricQuantumCircuit:
        ret = LinearMappedParametricQuantumCircuit(
            circuit.qubit_count, circuit.cbit_count
        )
        ret._param_mapping = LinearParameterMapping(circuit.param_mapping.in_params)
        pmap = circuit.param_mapping.mapping

        gates = []
        for gate, param in circuit.primitive_circuit().gates_and_params:
            if isinstance(gate, QuantumGate):
                gates.append(gate)
            else:
                if gates:
                    cc = QuantumCircuit(circuit.qubit_count, gates=gates)
                    ret.extend(self._transpiler(cc).gates)
                    gates = []

                if param is None:
                    raise ValueError("Parametric gate with no Parameter: {gate}")

                if gate.name == gate_names.ParametricRX:
                    ret.add_ParametricRX_gate(gate.target_indices[0], pmap[param])
                elif gate.name == gate_names.ParametricRY:
                    ret.add_ParametricRY_gate(gate.target_indices[0], pmap[param])
                elif gate.name == gate_names.ParametricRZ:
                    ret.add_ParametricRZ_gate(gate.target_indices[0], pmap[param])
                elif gate.name == gate_names.ParametricPauliRotation:
                    ret.add_ParametricPauliRotation_gate(
                        gate.target_indices, gate.pauli_ids, pmap[param]
                    )
                else:
                    raise ValueError(f"Unsupported parametric gate: {gate}")

        if gates:
            cc = QuantumCircuit(circuit.qubit_count, gates=gates)
            ret.extend(self._transpiler(cc).gates)

        return ret


class ParametricSequentialTranspiler(ParametricCircuitTranspilerProtocol):
    def __init__(self, transpilers: Sequence[ParametricCircuitTranspiler]):
        self._transpilers = transpilers

    def __call__(
        self, circuit: ParametricQuantumCircuitProtocol
    ) -> ParametricQuantumCircuitProtocol:
        for transpiler in self._transpilers:
            circuit = transpiler(circuit)
        return circuit
