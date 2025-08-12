# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import ceil, log2

import networkx as nx

from quri_parts.backend.device import DeviceProperty, GateProperty, QubitProperty
from quri_parts.backend.units import TimeValue
from quri_parts.circuit import gate_names
from quri_parts.circuit.transpile import (
    ParametricPauliRotationDecomposeTranspiler,
    ParametricRX2RZHTranspiler,
    ParametricRY2RZHTranspiler,
    ParametricSequentialTranspiler,
    ParametricTranspiler,
    STARSetTranspiler,
)


def generate_device_property(
    logical_qubit_count: int,
    qec_cycle: TimeValue,
    logical_error_rate: float,
    delta_sk: float,
    clifford_gate_cycles: int = 1,
    t_gate_cycles: int = 1,
) -> DeviceProperty:
    """Generate DeviceInfo object for Clifford + T architecture devices.

    Args:
        logical_qubit_count: Number of logical qubits.
        qec_cycle: Time duration of each syndrome measurement for quantum
            error correction (without code distance dependency).
        logical_error_rate: Logical error rate per QEC cycle.
        delta_sk: Required accuracy of sk decompositon of each rotation gate.
        clifford_gate_cycles: QEC cycles for each logical Clifford gate operation.
        t_gate_cycles: QEC cycles for each T gate operation.

    Returns:
        DeviceProperty object representing the target abstract FTQC architecture device.

    References:
        https://arxiv.org/abs/2303.13181
    """

    t_counts_for_rz = ceil(3 * log2(1 / delta_sk))
    rz_gate_cycles = t_gate_cycles * t_counts_for_rz

    qubits = list(range(logical_qubit_count))
    qubit_properties = {q: QubitProperty() for q in qubits}

    clifford_gate_time = TimeValue(
        value=clifford_gate_cycles * qec_cycle.value, unit=qec_cycle.unit
    )
    gate_properties = [
        GateProperty(
            gate_names.H,
            [],
            gate_error=0.0,
            gate_time=clifford_gate_time,
        ),
        GateProperty(
            gate_names.S,
            [],
            gate_error=0.0,
            gate_time=clifford_gate_time,
        ),
        GateProperty(
            gate_names.T,
            [],
            gate_error=0.0,
            gate_time=TimeValue(
                value=t_gate_cycles * qec_cycle.value, unit=qec_cycle.unit
            ),
        ),
        GateProperty(
            gate_names.CNOT,
            [],
            gate_error=0.0,
            gate_time=clifford_gate_time,
        ),
        GateProperty(
            gate_names.RZ,
            [],
            gate_error=delta_sk,
            gate_time=TimeValue(
                value=rz_gate_cycles * qec_cycle.value, unit=qec_cycle.unit
            ),
        ),
        GateProperty(
            gate_names.ParametricRZ,
            [],
            gate_error=delta_sk,
            gate_time=TimeValue(
                value=rz_gate_cycles * qec_cycle.value, unit=qec_cycle.unit
            ),
        ),
    ]

    trans = STARSetTranspiler()
    param_trans = ParametricSequentialTranspiler(
        [
            ParametricPauliRotationDecomposeTranspiler(),
            ParametricRX2RZHTranspiler(),
            ParametricRY2RZHTranspiler(),
            ParametricTranspiler(trans),
        ]
    )

    return DeviceProperty(
        qubit_count=logical_qubit_count,
        qubits=qubits,
        qubit_graph=nx.complete_graph(logical_qubit_count),
        qubit_properties=qubit_properties,
        native_gates=[
            gate_names.H,
            gate_names.S,
            gate_names.T,
            gate_names.CNOT,
        ],
        gate_properties=gate_properties,
        background_error=(logical_error_rate, qec_cycle),
        analyze_transpiler=trans,
        analyze_parametric_transpiler=param_trans,
    )
