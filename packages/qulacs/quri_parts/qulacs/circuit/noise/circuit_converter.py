# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools as it
from collections import defaultdict
from collections.abc import MutableMapping, MutableSequence, Sequence
from typing import Any, Callable, Mapping, Type, TypeVar

import qulacs
from typing_extensions import TypeAlias

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.noise import (
    AmplitudeDampingNoise,
    BitFlipNoise,
    BitPhaseFlipNoise,
    DepolarizingNoise,
    GateNoiseInstruction,
    GeneralDepolarizingNoise,
    KrausNoise,
    NoiseModel,
    PauliNoise,
    PhaseAmplitudeDampingNoise,
    PhaseDampingNoise,
    PhaseFlipNoise,
    ProbabilisticNoise,
    QubitNoisePair,
    ResetNoise,
    ThermalRelaxationNoise,
)
from quri_parts.qulacs.circuit import convert_gate

from .gate_converter import (
    create_kraus_gate,
    create_pauli_noise_gate,
    create_probabilistic_gate,
)

T = TypeVar("T", bound=GateNoiseInstruction)
GateNoiseConverter: TypeAlias = Callable[[Sequence[int], T], qulacs.QuantumGateBase]
GateConverterMapping: TypeAlias = Mapping[
    Type[GateNoiseInstruction], GateNoiseConverter[Any]
]

_gate_converter_mapping: GateConverterMapping = {
    ResetNoise: create_kraus_gate,
    PhaseDampingNoise: create_kraus_gate,
    AmplitudeDampingNoise: create_kraus_gate,
    PhaseAmplitudeDampingNoise: create_kraus_gate,
    ThermalRelaxationNoise: create_kraus_gate,
    PauliNoise: create_pauli_noise_gate,
    GeneralDepolarizingNoise: create_pauli_noise_gate,
    KrausNoise: create_kraus_gate,
    ProbabilisticNoise: create_probabilistic_gate,
}

_single_qubit_single_parameter_noise_mapping: Mapping[
    Type[GateNoiseInstruction], Callable[[int, float], qulacs.QuantumGateBase]
] = {
    BitFlipNoise: qulacs.gate.BitFlipNoise,
    PhaseFlipNoise: qulacs.gate.DephasingNoise,
    BitPhaseFlipNoise: qulacs.gate.IndependentXZNoise,
    DepolarizingNoise: qulacs.gate.DepolarizingNoise,
}


def convert_noise_to_gate(
    qubits: Sequence[int], noise: GateNoiseInstruction
) -> qulacs.QuantumGateBase:
    """Create :class:`qulacs.QuantumGateBase` representing the noise from the
    target qubit indices and :class:`GateNoiseInstruction`.

    Args:
        qubits: Sequence of qubit indices to which the noise is applied.
        noise: Noise representation as :class:`GateNoiseInstruction`.
    """

    noise_type = type(noise)

    if noise_type in _single_qubit_single_parameter_noise_mapping:
        return _single_qubit_single_parameter_noise_mapping[noise_type](
            qubits[0], noise.params[0]
        )
    elif noise_type in _gate_converter_mapping:
        return _gate_converter_mapping[noise_type](qubits, noise)
    else:
        raise NotImplementedError(f"Unsupported NoiseInstruction type: {noise_type}")


def convert_circuit_with_noise_model(
    circuit: NonParametricQuantumCircuit,
    noise_model: NoiseModel,
) -> qulacs.QuantumCircuit:
    """Apply the noise model while converting
    :class:`NonParametricQuantumCircuit` to
    :class:`qulacs.QuantumCircuit`.

    After conversion, several gates representing noises are inserted into
    :class:`qulacs.QuantumCircuit` depending on ``noise_model``.

    Args:
        circuit: :class:`NonParametricQuantumCircuit`, which is converted to
            :class:`qulacs.QuantumCircuit`.
        noise_model: :class:`NoiseModel` instance to be applied.
    """

    qulacs_circuit = qulacs.QuantumCircuit(circuit.qubit_count)
    resolvers = [c.create_resolver() for c in noise_model.noises_for_circuit()]

    depths: MutableMapping[int, int] = defaultdict(int)
    depth = 0

    for i, gate in enumerate(circuit.gates):
        gate_qubits = tuple(gate.control_indices) + tuple(gate.target_indices)

        # Circuit noises for depth.
        depth_noises: MutableSequence[QubitNoisePair] = []
        depth = 1 + max(depths[q] for q in gate_qubits)
        for q in gate_qubits:
            for ci in resolvers:
                depth_noises.extend(
                    ci.noises_for_depth(q, range(depths[q], depth), circuit)
                )
            depths[q] = depth  # Update depth.
        for qubits, gate_noise in depth_noises:
            qulacs_circuit.add_gate(convert_noise_to_gate(qubits, gate_noise))

        qulacs_gate = convert_gate(gate)
        qulacs_circuit.add_gate(qulacs_gate)

        noises = it.chain(
            # Gate noises.
            noise_model.noises_for_gate(gate),
            # Circuit noises for gate.
            *[ci.noises_for_gate(gate, i, circuit) for ci in resolvers],
        )
        for qubits, gate_noise in noises:
            qulacs_circuit.add_gate(convert_noise_to_gate(qubits, gate_noise))

    # Circuit noises for depth at the end of the circuit.
    depth_noises = []
    max_depth = max(depths.values())
    for q in range(circuit.qubit_count):
        for ci in resolvers:
            depth_noises.extend(
                ci.noises_for_depth(q, range(depths[q], max_depth + 1), circuit)
            )
    for qubits, gate_noise in depth_noises:
        qulacs_circuit.add_gate(convert_noise_to_gate(qubits, gate_noise))

    return qulacs_circuit
