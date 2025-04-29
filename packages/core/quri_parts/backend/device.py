# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from quri_parts.circuit import QuantumGate
from quri_parts.circuit.gate_names import is_parametric_gate_name
from quri_parts.circuit.noise import NoiseModel
from quri_parts.circuit.transpile import CircuitTranspiler, ParametricCircuitTranspiler

from .units import FrequencyValue, TimeValue


@dataclass(frozen=True)
class QubitProperty:
    """Noise property of a qubit.

    Args:
        T1 (TimeValue, optional): t1 relaxation time
        T2 (TimeValue, optional): t2 relaxation time
        frequency (FrequencyValue,optional): qubit frequency
        prob_meas0_on_state1 (float, optional): readout error probability of
            measuring 0 when the state is 1
        prob_meas1_on_state0 (float, optional): readout error probability of
            measuring 1 when the state is 0
        readout_time (TimeValue, optional): time duration on measurement.
        name (str, optional): name of the qubit
    """

    T1: Optional[TimeValue] = None
    T2: Optional[TimeValue] = None
    frequency: Optional[FrequencyValue] = None
    prob_meas0_on_state1: Optional[float] = None
    prob_meas1_on_state0: Optional[float] = None
    readout_time: Optional[TimeValue] = None
    name: Optional[str] = None


@dataclass(frozen=True)
class GateProperty:
    """Noise property of a gate.

    Args:
        gate (str): gate name
        qubits (Sequence[int]): target qubits for the gate. The order is control_index0,
            control_index1, ..., target_index0, ...
        gate_error (float, optional): 1 - fidelity of the gate operation
        gate_time (float, optional): time duration of the gate operation
        name (str, optional): name of the gate
    """

    gate: str
    qubits: Sequence[int]
    gate_error: Optional[float] = None
    gate_time: Optional[TimeValue] = None
    name: Optional[str] = None


@dataclass
class DeviceProperty:
    """Stores properties of a quantum device for circuit cost estimation.

    Args:
        qubit_count (int): Number of qubits.
        qubits (Sequence[int]): Qubit indices.
        qubit_graph (newtorkx.Graph): Topology of qubit connections.
        qubit_properties (Mapping[int, QubitProperty]): Mapping from qubit index to
            QubitProperty.
        native_gates (Collection[str]): Names of supported gates.
        gate_properties (Collection[GateProperty]): Collection of GateProperty.
        physical_qubit_count: (int, optional): Number of physical qubits.
        background_error: (tuple[float, TimeValue], optional): The errors that
            occur with respect to the passage of time for each qubit, regardless
            of the application of gates, etc.  It must be given together with the
            time unit.
        name (str, optional): Name of the device.
        provider (str, optional): Provider of the device.
        transpiler (CircuitTranspiler, optional): CircuitTranspiler for converting
            to an instruction sequence supported by the target device.
        parametric_transpiler (ParametricCircuitTranspiler, optional):
            ParametricCircuitTranspiler for converting to an instruction sequence
            supported by the target device.
        noise_model (NoiseModel, optional): Noise models that reproduce device
            behaviour.
    """

    qubit_count: int
    qubits: Sequence[int]
    qubit_graph: nx.Graph
    qubit_properties: Mapping[int, QubitProperty]
    native_gates: Sequence[str]
    _gate_properties: Mapping[tuple[str, tuple[int, ...]], GateProperty]
    physical_qubit_count: Optional[int] = None
    background_error: Optional[tuple[float, TimeValue]] = None
    name: Optional[str] = None
    provider: Optional[str] = None

    transpiler: Optional[CircuitTranspiler] = None
    parametric_transpiler: Optional[ParametricCircuitTranspiler] = None
    analyze_transpiler: Optional[CircuitTranspiler] = None
    analyze_parametric_transpiler: Optional[ParametricCircuitTranspiler] = None
    noise_model: Optional[NoiseModel] = None

    def __init__(
        self,
        qubit_count: int,
        qubits: Sequence[int],
        qubit_graph: nx.Graph,
        qubit_properties: Mapping[int, QubitProperty],
        native_gates: Collection[str],
        gate_properties: Collection[GateProperty],
        physical_qubit_count: Optional[int] = None,
        background_error: Optional[tuple[float, TimeValue]] = None,
        name: Optional[str] = None,
        provider: Optional[str] = None,
        transpiler: Optional[CircuitTranspiler] = None,
        parametric_transpiler: Optional[ParametricCircuitTranspiler] = None,
        analyze_transpiler: Optional[CircuitTranspiler] = None,
        analyze_parametric_transpiler: Optional[ParametricCircuitTranspiler] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> None:
        self.qubit_count = qubit_count
        self.qubits = tuple(qubits)
        self.qubit_graph = qubit_graph
        self.qubit_properties = qubit_properties
        self.native_gates = tuple(native_gates)
        self._gate_properties = {
            (prop.gate, tuple(prop.qubits)): prop for prop in gate_properties
        }
        self.physical_qubit_count = physical_qubit_count
        self.background_error = background_error
        self.name = name
        self.provider = provider
        self.transpiler = transpiler
        self.parametric_transpiler = parametric_transpiler
        self.analyze_transpiler = analyze_transpiler
        self.analyze_parametric_transpiler = analyze_parametric_transpiler
        self.noise_model = noise_model

    def gate_property(self, quantum_gate: QuantumGate) -> GateProperty:
        """Returns GateProperty of the device corresponding to the given
        QuantumGate.

        If the given quantum gate is not specified with qubits, the
        GateProperty for the kind of the quantum gate is searched.
        """

        if is_parametric_gate_name(quantum_gate.name):
            raise ValueError(f"Unsupported gate kind: {quantum_gate.name}")
        gate = (
            quantum_gate.name,
            tuple(quantum_gate.control_indices) + tuple(quantum_gate.target_indices),
        )
        gate = gate if gate in self._gate_properties else (gate[0], ())
        return self._gate_properties[gate]
