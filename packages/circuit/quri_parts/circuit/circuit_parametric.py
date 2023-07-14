# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Optional, Protocol, Union, runtime_checkable

from quri_parts.circuit import gate_names

from .circuit import (
    GateSequence,
    ImmutableQuantumCircuit,
    MutableQuantumCircuitProtocol,
    NonParametricQuantumCircuit,
    QuantumCircuit,
    QuantumCircuitProtocol,
    is_gate_sequence,
)
from .gate import ParametricQuantumGate, QuantumGate
from .gates import (
    RX,
    RY,
    RZ,
    ParametricPauliRotation,
    ParametricRX,
    ParametricRY,
    ParametricRZ,
    PauliRotation,
)
from .parameter import Parameter
from .parameter_mapping import LinearParameterMapping, ParameterMapping


@runtime_checkable
class UnboundParametricQuantumCircuitProtocol(QuantumCircuitProtocol, Protocol):
    """Interface protocol for a quantum circuit containing unbound (i.e. not
    assigned values) parameters.

    For circuit execution, the parameters need to be assigned concrete
    values by :meth:`~bind_parameters` method.
    """

    @abstractmethod
    def bind_parameters(
        self, params: Sequence[float]
    ) -> "ImmutableBoundParametricQuantumCircuit":
        """Returns a new circuit with the parameters assigned concrete values.

        This method does not modify self but returns a newly created
        circuit.
        """
        ...

    @abstractmethod
    def freeze(self) -> "UnboundParametricQuantumCircuitProtocol":
        """Returns a \"freezed\" version of itself.

        The \"freezed\" version is an immutable object and can be reused
        safely without copying.
        """
        ...

    @abstractmethod
    def get_mutable_copy(self) -> "MutableUnboundParametricQuantumCircuitProtocol":
        """Returns a copy of itself that can be modified.

        Use this method when you want to get a new circuit based on an
        existing circuit but don't want to modify it.
        """
        ...

    @abstractmethod
    def primitive_circuit(self) -> "ImmutableUnboundParametricQuantumCircuit":
        r"""Returns the parametric circuit where each gate has an independent
        parameter.

        Note that some parametric circuit,
        e.g. :class:`LinearMappedUnboundParametricQuantumCircuit`, can have non-trivial
        mapping of the parameters. In this "primitive circuit", however,
        gate parameters are treated as independent,
        even if those in the original circuit depend on the same parameters.
        For example, if the parametric circuit is defined as:

        .. math::
            \begin{align}
                U_1(f(\theta_1, \theta_2)) U_2(g(\theta_1, \theta_2))
            \end{align}

        the primitive circuit should be as the following:

        .. math::
            \begin{align}
                U_1(\psi_1) U_2(\psi_2)
            \end{align}

        where U1, U2 are rotation gates and f, g are parameter mappings.
        """
        ...

    @abstractproperty
    def parameter_count(self) -> int:
        ...

    @abstractproperty
    def gates(self) -> Sequence[Union[QuantumGate, ParametricQuantumGate]]:
        """Returns the gate sequence of the circuit."""
        ...

    @abstractproperty
    def has_trivial_parameter_mapping(self) -> bool:
        """Returns if the input parameters are used for parametric gates
        without any conversions.

        Note that some parametric circuit,
        e.g. :class:`LinearMappedUnboundParametricQuantumCircuit`, can have non-trivial
        mapping of the parameters.
        """
        ...

    @abstractproperty
    def param_mapping(self) -> ParameterMapping:
        """Returns the parameter mapping of the circuit."""
        ...

    def __add__(
        self, gates: Union[GateSequence, "UnboundParametricQuantumCircuitProtocol"]
    ) -> "UnboundParametricQuantumCircuitProtocol":
        """Returns a new combined circuit with the given gates added."""
        ...

    def __radd__(
        self, gates: Union[GateSequence, "UnboundParametricQuantumCircuitProtocol"]
    ) -> "UnboundParametricQuantumCircuitProtocol":
        """Returns a new combined circuit with the given gates added."""
        ...


class MutableUnboundParametricQuantumCircuitProtocol(
    UnboundParametricQuantumCircuitProtocol, MutableQuantumCircuitProtocol, Protocol
):
    ...


class UnboundParametricQuantumCircuitBase(UnboundParametricQuantumCircuitProtocol, ABC):
    """A base class for quantum circuits having parametric gates with unbound
    parameters.

    This class is for parametric circuits where all parametric gates have independent
    parameters, i.e., no two parametric gates share the same parameter. If you want to
    make dependency between parameters, see
    :class:`~LinearMappedUnboundParametricQuantumCircuitBase`.
    """

    _gates: Sequence[
        Union[tuple[QuantumGate, None], tuple[ParametricQuantumGate, Parameter]]
    ]

    @property
    def depth(self) -> int:
        max_layers_each_qubit = [0] * self.qubit_count
        for gate, _ in self._gates:
            acting_ids = list(gate.control_indices) + list(gate.target_indices)
            max_layers = max(max_layers_each_qubit[index] for index in acting_ids)
            for index in acting_ids:
                max_layers_each_qubit[index] = max_layers + 1
        return max(max_layers_each_qubit)

    @property
    def gates(self) -> Sequence[Union[QuantumGate, ParametricQuantumGate]]:
        return [gate for gate, _ in self._gates]

    @property
    def gates_and_params(
        self,
    ) -> Sequence[
        Union[tuple[QuantumGate, None], tuple[ParametricQuantumGate, Parameter]]
    ]:
        """Returns the sequence of the tuples of gate and it's parameter."""
        return tuple(self._gates)

    @property
    def has_trivial_parameter_mapping(self) -> bool:
        return True

    @property
    def param_mapping(self) -> LinearParameterMapping:
        return LinearParameterMapping(
            self._params, self._params, dict(zip(self._params, self._params))
        )

    def primitive_circuit(self) -> "ImmutableUnboundParametricQuantumCircuit":
        return self.freeze()

    def get_mutable_copy(self) -> "UnboundParametricQuantumCircuit":
        circuit = UnboundParametricQuantumCircuit(self.qubit_count, self.cbit_count)
        circuit._gates = list(self._gates)
        return circuit

    def combine(
        self, gates: Union[GateSequence, "UnboundParametricQuantumCircuitBase"]
    ) -> "UnboundParametricQuantumCircuit":
        """Create a new parametric circuit with itself and the given gates or
        an unbound parametric circuit combined.

        All parameters in self and the other are treated as different
        ones. I.e. even if those two share the same instance of a
        parameter, it is copied into two independent parameters.
        """
        circuit = UnboundParametricQuantumCircuit(self.qubit_count, self.cbit_count)
        circuit.extend(self)
        circuit.extend(gates)
        return circuit

    @abstractmethod
    def freeze(self) -> "ImmutableUnboundParametricQuantumCircuit":
        ...

    def bind_parameters(
        self, params: Sequence[float]
    ) -> "ImmutableBoundParametricQuantumCircuit":
        param_list = self._params
        if len(params) != len(param_list):
            raise ValueError(
                f"Passed value count ({len(params)}) does not match parameter count"
                f"({len(param_list)})."
            )
        return ImmutableBoundParametricQuantumCircuit(
            self, dict(zip(param_list, params))
        )

    @property
    def _params(self) -> Sequence[Parameter]:
        return [p for _, p in self._gates if p is not None]

    @property
    def parameter_count(self) -> int:
        return len(self._params)

    def __add__(
        self,
        gates: Union[GateSequence, UnboundParametricQuantumCircuitProtocol],
    ) -> "UnboundParametricQuantumCircuit":
        if isinstance(gates, UnboundParametricQuantumCircuitBase):
            return self.combine(gates)
        elif is_gate_sequence(gates):
            return self.combine(gates)
        else:
            return NotImplemented

    def __radd__(
        self,
        gates: Union[GateSequence, UnboundParametricQuantumCircuitProtocol],
    ) -> "UnboundParametricQuantumCircuit":
        if isinstance(gates, UnboundParametricQuantumCircuitBase) or is_gate_sequence(
            gates
        ):
            combined_circuit = UnboundParametricQuantumCircuit(self.qubit_count)
            combined_circuit.extend(gates)
            combined_circuit.extend(self)
            return combined_circuit
        else:
            return NotImplemented


class UnboundParametricQuantumCircuit(
    UnboundParametricQuantumCircuitBase,
    MutableUnboundParametricQuantumCircuitProtocol,
):
    """A mutable unbound parametric quantum circuit.

    This class is for parametric circuits where all parametric gates have independent
    parameters, i.e., no two parametric gates share the same parameter. If you want to
    make dependency between parameters, see
    :class:`~LinearMappedUnboundParametricQuantumCircuit`.
    """

    def __init__(self, qubit_count: int, cbit_count: int = 0):
        self._qubit_count = qubit_count
        self._gates: list[
            Union[tuple[QuantumGate, None], tuple[ParametricQuantumGate, Parameter]]
        ] = []
        self._cbit_count = cbit_count

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnboundParametricQuantumCircuitBase):
            return False
        return self.freeze() == other.freeze()

    @property
    def qubit_count(self) -> int:
        return self._qubit_count

    @property
    def cbit_count(self) -> int:
        return self._cbit_count

    def add_gate(self, gate: QuantumGate, gate_index: Optional[int] = None) -> None:
        if max([*gate.control_indices, *gate.target_indices]) >= self.qubit_count:
            raise ValueError(
                "The indices of the gate applied must be smaller than qubit_count"
            )
        if gate_index:
            self._gates.insert(gate_index, (gate, None))
        else:
            self._gates.append((gate, None))

    def add_ParametricRX_gate(self, qubit_index: int) -> Parameter:
        """Add a parametric RX gate to the circuit."""
        if qubit_index >= self.qubit_count:
            raise ValueError(
                "The indices of the gate applied must be smaller than qubit_count"
            )
        p = Parameter()
        self._gates.append((ParametricRX(qubit_index), p))
        return p

    def add_ParametricRY_gate(self, qubit_index: int) -> Parameter:
        if qubit_index >= self.qubit_count:
            raise ValueError(
                "The indices of the gate applied must be smaller than qubit_count"
            )
        """Add a parametric RY gate to the circuit."""
        p = Parameter()
        self._gates.append((ParametricRY(qubit_index), p))
        return p

    def add_ParametricRZ_gate(self, qubit_index: int) -> Parameter:
        if qubit_index >= self.qubit_count:
            raise ValueError(
                "The indices of the gate applied must be smaller than qubit_count"
            )
        """Add a parametric RZ gate to the circuit."""
        p = Parameter()
        self._gates.append((ParametricRZ(qubit_index), p))
        return p

    def add_ParametricPauliRotation_gate(
        self, target_indices: Sequence[int], pauli_ids: Sequence[int]
    ) -> Parameter:
        """Add a parametric Pauli rotation gate to the circuit."""
        if max(target_indices) >= self.qubit_count:
            raise ValueError(
                "The indices of the gate applied must be smaller than qubit_count"
            )
        p = Parameter()
        self._gates.append((ParametricPauliRotation(target_indices, pauli_ids), p))
        return p

    def extend(
        self, gates: Union[GateSequence, UnboundParametricQuantumCircuitBase]
    ) -> None:
        """Extend the parametric circuit with given gates or an unbound
        parametric circuit.

        All parameters in self and the other are treated as different
        ones. I.e. even if those two share the same instance of a
        parameter, it is copied into two independent parameters.
        """
        if isinstance(gates, UnboundParametricQuantumCircuitBase):
            self._gates.extend(gates._gates)
        else:
            if isinstance(gates, NonParametricQuantumCircuit):
                gates = gates.gates
            for g in gates:
                self.add_gate(g)

    def __iadd__(
        self,
        gates: Union[GateSequence, UnboundParametricQuantumCircuitProtocol],
    ) -> "UnboundParametricQuantumCircuit":
        if isinstance(gates, UnboundParametricQuantumCircuitBase):
            self.extend(gates)
            return self
        elif is_gate_sequence(gates):
            self.extend(gates)
            return self
        else:
            return NotImplemented

    def freeze(self) -> "ImmutableUnboundParametricQuantumCircuit":
        return ImmutableUnboundParametricQuantumCircuit(self)


class ImmutableUnboundParametricQuantumCircuit(UnboundParametricQuantumCircuitBase):
    """An immutable unbound parametric quantum circuit.

    This class is for parametric circuits where all parametric gates have independent
    parameters, i.e., no two parametric gates share the same parameter. If you want to
    make dependency between parameters, see
    :class:`~ImmutableLinearMappedUnboundParametricQuantumCircuit`.
    """

    def __init__(self, circuit: UnboundParametricQuantumCircuitBase):
        self._qubit_count = circuit.qubit_count
        self._gates = tuple(circuit._gates)
        self._cbit_count = circuit.cbit_count

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnboundParametricQuantumCircuitBase):
            return False
        f = other.freeze()
        return (self.qubit_count, self.cbit_count, self._gates) == (
            f.qubit_count,
            f.cbit_count,
            f._gates,
        )

    @property
    def qubit_count(self) -> int:
        return self._qubit_count

    @property
    def cbit_count(self) -> int:
        return self._cbit_count

    def freeze(self) -> "ImmutableUnboundParametricQuantumCircuit":
        return self


class ImmutableBoundParametricQuantumCircuit(ImmutableQuantumCircuit):
    """An immutable \"bound\" parametric quantum circuit, i.e. a parametric
    circuit with its parameters assigned concrete values."""

    def __init__(
        self,
        circuit: UnboundParametricQuantumCircuitBase,
        parameter_map: Mapping[Parameter, float],
    ):
        super().__init__(QuantumCircuit(circuit.qubit_count, circuit.cbit_count))
        self.unbound_param_circuit = circuit.freeze()
        self.parameter_map = dict(parameter_map)

    @cached_property
    def gates(self) -> Sequence[QuantumGate]:
        def map_param_gate(
            gate: Union[QuantumGate, ParametricQuantumGate], param: Optional[Parameter]
        ) -> QuantumGate:
            if isinstance(gate, QuantumGate):
                return gate
            elif isinstance(gate, ParametricQuantumGate) and param is not None:
                value = self.parameter_map[param]
                if gate.name == gate_names.ParametricRX:
                    return RX(gate.target_indices[0], value)
                elif gate.name == gate_names.ParametricRY:
                    return RY(gate.target_indices[0], value)
                elif gate.name == gate_names.ParametricRZ:
                    return RZ(gate.target_indices[0], value)
                elif gate.name == gate_names.ParametricPauliRotation:
                    return PauliRotation(gate.target_indices, gate.pauli_ids, value)
                else:
                    raise ValueError("Unknown gate")
            else:
                raise ValueError("ParametricQuantumGate must have a parameter.")

        return tuple(map_param_gate(*t) for t in self.unbound_param_circuit._gates)

    def freeze(self) -> "ImmutableBoundParametricQuantumCircuit":
        return self
