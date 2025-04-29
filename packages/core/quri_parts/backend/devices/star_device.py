# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import ceil
from typing import cast

import networkx as nx

from quri_parts.backend.device import DeviceProperty, GateProperty, QubitProperty
from quri_parts.backend.units import TimeValue
from quri_parts.circuit import gate_names, noise
from quri_parts.circuit.transpile import (
    ParametricPauliRotationDecomposeTranspiler,
    ParametricRX2RZHTranspiler,
    ParametricRY2RZHTranspiler,
    ParametricSequentialTranspiler,
    ParametricTranspiler,
    STARSetTranspiler,
)


def generate_device_property(
    qubit_count: int,
    code_distance: int,
    qec_cycle: TimeValue,
    physical_error_rate: float = 0.0,
    data_total_qubit_ratio: float = 4.0,
) -> DeviceProperty:
    """Generate DeviceInfo object for STAR architecture devices.

    Args:
        qubit_count: Number of logical qubits.
        physical_error_rate: Error rate of physical qubit operations.
        code_distance: Code distance of the quantum error correction code.
        qec_cycle: Time duration of each syndrome measurement for quantum
            error correction (without code distance dependency).
        data_total_qubit_ratio: Ratio of number of total logical qubits to
            number of data logical qubits.

    Returns:
        DeviceInfo object representing the target STAR architecture device.

    References:
        https://arxiv.org/abs/2303.13181
    """

    # cf. https://arxiv.org/abs/2303.13181 (pp.11-14)
    def logical_error_model(ci: float, pthi: float, p: float, d: int) -> float:
        return cast(float, ci * (p / pthi) ** ((d + 1.0) / 2.0))

    def logical_error_round(p: float, d: int) -> float:
        plz = logical_error_model(ci=0.067976, pthi=0.0038510, p=p, d=d)
        plx = logical_error_model(ci=0.081997, pthi=0.0041612, p=p, d=d)
        return min(plz + plx, 1.0)

    logical_fidelity_round = 1.0 - logical_error_round(
        p=physical_error_rate, d=code_distance
    )

    rus_steps = 2

    qubits = list(range(qubit_count))
    qubit_properties = {q: QubitProperty() for q in qubits}
    gate_properties = [
        GateProperty(
            gate_names.H,
            (),
            gate_error=0.0,
            gate_time=TimeValue(
                value=3.0 * code_distance * qec_cycle.value, unit=qec_cycle.unit
            ),
        ),
        GateProperty(
            gate_names.S,
            [],
            gate_error=0.0,
            # TODO Confirm latency value
            gate_time=TimeValue(
                value=2.0 * code_distance * qec_cycle.value, unit=qec_cycle.unit
            ),
        ),
        GateProperty(
            gate_names.CNOT,
            (),
            gate_error=0.0,
            gate_time=TimeValue(
                value=2.0 * code_distance * qec_cycle.value, unit=qec_cycle.unit
            ),
        ),
        GateProperty(
            gate_names.RZ,
            (),
            gate_error=1.0 - (1.0 - 2.0 * physical_error_rate / 15.0) ** rus_steps,
            gate_time=TimeValue(
                value=2.0 * rus_steps * code_distance * qec_cycle.value,
                unit=qec_cycle.unit,
            ),
        ),
        GateProperty(
            gate_names.ParametricRZ,
            (),
            gate_error=1.0 - (1.0 - 2.0 * physical_error_rate / 15.0) ** rus_steps,
            gate_time=TimeValue(
                value=2.0 * rus_steps * code_distance * qec_cycle.value,
                unit=qec_cycle.unit,
            ),
        ),
    ]
    qec_fidelity_per_qec_cycle = logical_fidelity_round**data_total_qubit_ratio

    total_patches = ceil(qubit_count * data_total_qubit_ratio)
    # 2d^2 physical qubits per single patch, including syndrome measurement qubits
    physical_qubit_count = (2 * code_distance**2) * total_patches

    p1 = 1.0 - (1.0 - 2.0 * physical_error_rate / 15.0) ** rus_steps
    noise_model = noise.NoiseModel(
        [noise.PhaseFlipNoise(error_prob=p1, target_gates=[gate_names.RZ])]
    )

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
        qubit_count=qubit_count,
        qubits=qubits,
        qubit_graph=nx.complete_graph(qubit_count),
        qubit_properties=qubit_properties,
        native_gates=[
            gate_names.H,
            gate_names.S,
            gate_names.CNOT,
            gate_names.RZ,
        ],
        gate_properties=gate_properties,
        physical_qubit_count=physical_qubit_count,
        background_error=(1.0 - qec_fidelity_per_qec_cycle, qec_cycle),
        transpiler=trans,
        parametric_transpiler=param_trans,
        noise_model=noise_model,
    )
