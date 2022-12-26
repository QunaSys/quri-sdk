# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from collections.abc import Iterable, Sequence
from typing import Optional, Union

from .circuit import GateSequence, NonParametricQuantumCircuit
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
from .gate import ParametricQuantumGate, QuantumGate
from .parameter_mapping import LinearParameterMapping, ParameterOrLinearFunction


class LinearMappedUnboundParametricQuantumCircuitBase(
    UnboundParametricQuantumCircuitProtocol, ABC
):
    """A base class for parametric quantum circuits where parameters of
    parametric gates are given by linear functions of circuit parameters."""

    _param_mapping: LinearParameterMapping
    _circuit: UnboundParametricQuantumCircuitBase

    @property
    def qubit_count(self) -> int:
        return self._circuit.qubit_count

    @property
    def parameter_count(self) -> int:
        return len(self._param_mapping.in_params)

    @property
    def depth(self) -> int:
        return self._circuit.depth

    @property
    def gates(self) -> Sequence[Union[QuantumGate, ParametricQuantumGate]]:
        return [gate for gate, _ in self._circuit._gates]

    @property
    def has_trivial_parameter_mapping(self) -> bool:
        return self._param_mapping.is_trivial_mapping

    @property
    def mapping_and_raw_circuit(
        self,
    ) -> tuple[LinearParameterMapping, ImmutableUnboundParametricQuantumCircuit]:
        return (self._param_mapping, self._circuit.freeze())

    def get_mutable_copy(self) -> "LinearMappedUnboundParametricQuantumCircuit":
        circuit = LinearMappedUnboundParametricQuantumCircuit(self.qubit_count)
        circuit._param_mapping = self._param_mapping
        circuit._circuit = self._circuit.get_mutable_copy()
        return circuit

    def combine(
        self,
        gates: Union[GateSequence, "LinearMappedUnboundParametricQuantumCircuitBase"],
    ) -> "LinearMappedUnboundParametricQuantumCircuit":
        circuit = LinearMappedUnboundParametricQuantumCircuit(self.qubit_count)
        circuit.extend(self)
        circuit.extend(gates)
        return circuit

    def __add__(
        self,
        gates: Union[GateSequence, "LinearMappedUnboundParametricQuantumCircuitBase"],
    ) -> "LinearMappedUnboundParametricQuantumCircuit":
        return self.combine(gates)

    def bind_parameters(
        self, params: Sequence[float]
    ) -> ImmutableBoundParametricQuantumCircuit:
        raw_param_vars = self._param_mapping.mapper(
            dict(zip(self._param_mapping.in_params, params))
        )
        return ImmutableBoundParametricQuantumCircuit(self._circuit, raw_param_vars)


class LinearMappedUnboundParametricQuantumCircuit(
    LinearMappedUnboundParametricQuantumCircuitBase,
    MutableUnboundParametricQuantumCircuitProtocol,
):
    """A mutable parametric quantum circuit where parameters of parametric
    gates are given by linear functions of circuit parameters."""

    def __init__(self, qubit_count: int):
        self._param_mapping = LinearParameterMapping()
        self._circuit: UnboundParametricQuantumCircuit = (
            UnboundParametricQuantumCircuit(qubit_count)
        )

    def add_parameters(self, *names: str) -> Sequence[Parameter]:
        """Add new parameters for the circuit.

        Newly created parameters are returned as a sequence. Before
        adding a parametric gate to the circuit, all used parameters
        should be added by this method or :meth:`~add_parameter`.
        """
        added = [Parameter(name) for name in names]
        self._param_mapping = self._param_mapping.with_data_updated(
            in_params_addition=added
        )
        return tuple(added)

    def add_parameter(self, name: str) -> Parameter:
        """Add a new parameter for the circuit.

        Before adding a parametric gate to the circuit, all used
        parameters should be added by this method or
        :meth:`~add_parametesr`.
        """
        return self.add_parameters(name)[0]

    def _check_param_exist(self, param: ParameterOrLinearFunction) -> None:
        params: Iterable[Parameter]
        if isinstance(param, Parameter):
            params = (param,)
        else:
            params = param.keys()

        for p in params:
            if p not in self._param_mapping.in_params and (p is not CONST):
                raise ValueError(
                    f"The given parameter {p} does not belong to this"
                    "LinearMappedUnboundParametricQuantumCircuit."
                )

    def add_gate(self, gate: QuantumGate, gate_index: Optional[int] = None) -> None:
        self._circuit.add_gate(gate, gate_index)

    def add_ParametricRX_gate(
        self, qubit_index: int, angle: ParameterOrLinearFunction
    ) -> None:
        """Add a parametric RX gate to the circuit."""
        self._check_param_exist(angle)
        param = self._circuit.add_ParametricRX_gate(qubit_index)
        self._param_mapping = self._param_mapping.with_data_updated(
            out_params_addition=(param,), mapping_update={param: angle}
        )

    def add_ParametricRY_gate(
        self, qubit_index: int, angle: ParameterOrLinearFunction
    ) -> None:
        """Add a parametric RY gate to the circuit."""
        self._check_param_exist(angle)
        param = self._circuit.add_ParametricRY_gate(qubit_index)
        self._param_mapping = self._param_mapping.with_data_updated(
            out_params_addition=(param,), mapping_update={param: angle}
        )

    def add_ParametricRZ_gate(
        self, qubit_index: int, angle: ParameterOrLinearFunction
    ) -> None:
        """Add a parametric RZ gate to the circuit."""
        self._check_param_exist(angle)
        param = self._circuit.add_ParametricRZ_gate(qubit_index)
        self._param_mapping = self._param_mapping.with_data_updated(
            out_params_addition=(param,), mapping_update={param: angle}
        )

    def add_ParametricPauliRotation_gate(
        self,
        qubit_indices: Sequence[int],
        pauli_ids: Sequence[int],
        angle: ParameterOrLinearFunction,
    ) -> None:
        """Add a parametric Pauli rotation gate to the circuit."""
        self._check_param_exist(angle)
        param = self._circuit.add_ParametricPauliRotation_gate(qubit_indices, pauli_ids)
        self._param_mapping = self._param_mapping.with_data_updated(
            out_params_addition=(param,), mapping_update={param: angle}
        )

    def extend(
        self,
        gates: Union[GateSequence, LinearMappedUnboundParametricQuantumCircuitBase],
    ) -> None:
        """Extend the parametric circuit with given gates or a linear mapped
        unbound parametric circuit.

        If the two linear mapped parametric circuit share the same
        parameters, they are treated as the same parameters, in contrast
        to the case of :class:`~UnboundParametricQuantumCircuit`.
        """
        if isinstance(gates, LinearMappedUnboundParametricQuantumCircuitBase):
            if self.qubit_count != gates.qubit_count:
                raise ValueError(
                    f"Qubit count not match (self={self.qubit_count}, "
                    f"other={gates.qubit_count})."
                )
            self._param_mapping = self._param_mapping.combine(gates._param_mapping)
            self._circuit.extend(gates._circuit)
        else:
            if isinstance(gates, NonParametricQuantumCircuit):
                gates = gates.gates
            for gate in gates:
                self.add_gate(gate)

    def freeze(self) -> "ImmutableLinearMappedUnboundParametricQuantumCircuit":
        return ImmutableLinearMappedUnboundParametricQuantumCircuit(self)


class ImmutableLinearMappedUnboundParametricQuantumCircuit(
    LinearMappedUnboundParametricQuantumCircuitBase
):
    """An immutable parametric quantum circuit where parameters of parametric
    gates are given by linear functions of circuit parameters."""

    def __init__(self, circuit: LinearMappedUnboundParametricQuantumCircuit):
        self._param_mapping = circuit._param_mapping
        self._circuit = circuit._circuit.freeze()

    def freeze(self) -> "ImmutableLinearMappedUnboundParametricQuantumCircuit":
        return self
