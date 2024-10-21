# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import Callable

import numpy as np

import quri_parts.circuit.gate_names as names
from quri_parts.circuit import QuantumGate, gates
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
    ResetNoise,
    ThermalRelaxationNoise,
)


def test_single_qubit_gate_mapping() -> None:
    noises = [BitFlipNoise(0.001, [0, 1, 2], [names.H, names.X, names.Z])]
    model = NoiseModel(noises)
    fs: Sequence[Callable[[int], QuantumGate]] = [gates.H, gates.X, gates.Z]
    for factory in fs:
        for qubit in [0, 1, 2]:
            qs, x = next(iter(model.noises_for_gate(factory(qubit))))
            assert qs == (qubit,)
            assert x is noises[0]


def test_two_qubit_gate_mapping() -> None:
    noises = [PhaseFlipNoise(0.002, [0, 1, 2], [names.CNOT, names.SWAP])]
    model = NoiseModel(noises)
    assert list(model.noises_for_gate(gates.CNOT(0, 1))) == [
        ((0,), noises[0]),
        ((1,), noises[0]),
    ]
    assert list(model.noises_for_gate(gates.SWAP(2, 3))) == [((2,), noises[0])]


def test_gate_noises_for_all() -> None:
    noises = [BitPhaseFlipNoise(0.003, [], [])]
    model = NoiseModel(noises)
    assert list(model.noises_for_gate(gates.Y(3))) == [((3,), noises[0])]
    assert list(model.noises_for_gate(gates.CNOT(5, 7))) == [
        ((5,), noises[0]),
        ((7,), noises[0]),
    ]


def test_gate_noises_for_all_qubits_specified_gates() -> None:
    noises = [DepolarizingNoise(0.004, [], [names.X, names.CNOT])]
    model = NoiseModel(noises)
    assert list(model.noises_for_gate(gates.X(1))) == [((1,), noises[0])]
    assert list(model.noises_for_gate(gates.CNOT(3, 4))) == [
        ((3,), noises[0]),
        ((4,), noises[0]),
    ]


def test_gate_noises_for_all_gates_specified_qubits() -> None:
    noises = [ResetNoise(0.001, 0.002, [1, 3], [])]
    model = NoiseModel(noises)
    assert list(model.noises_for_gate(gates.H(3))) == [((3,), noises[0])]
    assert list(model.noises_for_gate(gates.CNOT(1, 2))) == [((1,), noises[0])]


def test_multiple_noises() -> None:
    noises = [
        PhaseDampingNoise(0.005, [], [names.H, names.SWAP]),
        AmplitudeDampingNoise(0.004, 0.003, [2, 4], []),
        PhaseAmplitudeDampingNoise(0.002, 0.001, 0.002, [0], [names.H]),
        ThermalRelaxationNoise(
            0.003, 0.004, 0.005, 0.004, [1, 0], [names.X, names.CNOT]
        ),
        PauliNoise([[1]], [0.003], [], []),
    ]
    model = NoiseModel(noises)
    assert list(model.noises_for_gate(gates.H(0))) == [
        ((0,), noises[0]),
        ((0,), noises[2]),
        ((0,), noises[4]),
    ]
    assert list(model.noises_for_gate(gates.CNOT(1, 2))) == [
        ((2,), noises[1]),
        ((1,), noises[3]),
        ((1,), noises[4]),
        ((2,), noises[4]),
    ]
    assert list(model.noises_for_gate(gates.CNOT(1, 0))) == [
        ((1,), noises[3]),
        ((0,), noises[3]),
        ((1,), noises[4]),
        ((0,), noises[4]),
    ]
    assert list(model.noises_for_gate(gates.SWAP(2, 3))) == [
        ((2,), noises[0]),
        ((3,), noises[0]),
        ((2,), noises[1]),
        ((2,), noises[4]),
        ((3,), noises[4]),
    ]
    assert list(model.noises_for_gate(gates.X(3))) == [
        ((3,), noises[4]),
    ]
    assert list(model.noises_for_gate(gates.Y(4))) == [
        ((4,), noises[1]),
        ((4,), noises[4]),
    ]


def test_multi_qubit_pauli_noises() -> None:
    noises = [
        PauliNoise(
            [[1, 2], [2, 3], [3, 1]],
            [0.001, 0.002, 0.003],
            [2, 0],
            [names.CNOT, names.Pauli],
        ),
        GeneralDepolarizingNoise(0.004, 3, [1, 2, 3], [names.Pauli]),
    ]
    model = NoiseModel(noises)
    assert list(model.noises_for_gate(gates.CNOT(2, 0))) == [
        ((2, 0), noises[0]),
    ]
    assert list(model.noises_for_gate(gates.Pauli([3, 1, 2], [1, 2, 3]))) == [
        ((3, 1, 2), noises[1]),
    ]
    assert list(model.noises_for_gate(gates.Pauli([1, 2, 3], [1, 2, 3]))) == [
        ((1, 2, 3), noises[1]),
    ]


def test_multi_qubit_noises() -> None:
    kraus_list = list(np.random.random((4, 2, 2)))
    dense_matrix_list = list(np.random.random((3, 4, 4)))
    prob_list = [0.002, 0.004, 0.003]
    noises = [
        KrausNoise(kraus_list, [1, 2], [names.H, names.CNOT]),
        ProbabilisticNoise(dense_matrix_list, prob_list, [0, 3], [names.SWAP, names.X]),
    ]
    model = NoiseModel(noises)
    assert list(model.noises_for_gate(gates.H(1))) == [((1,), noises[0])]
    assert list(model.noises_for_gate(gates.CNOT(2, 1))) == [
        ((2,), noises[0]),
        ((1,), noises[0]),
    ]
    assert list(model.noises_for_gate(gates.CNOT(1, 3))) == [((1,), noises[0])]
    assert list(model.noises_for_gate(gates.SWAP(0, 3))) == [((0, 3), noises[1])]
    assert list(model.noises_for_gate(gates.SWAP(3, 0))) == [((3, 0), noises[1])]
    assert list(model.noises_for_gate(gates.X(3))) == []


def _kraus_list() -> Sequence[Sequence[Sequence[float]]]:
    p0, p1 = 0.005, 0.002
    return [
        [[np.sqrt(1 - p0 - p1), 0], [0, np.sqrt(1 - p0 - p1)]],
        [[np.sqrt(p0), 0], [0, 0]],
        [[0, np.sqrt(p0)], [0, 0]],
        [[0, 0], [np.sqrt(p1), 0]],
        [[0, 0], [0, np.sqrt(p1)]],
    ]


_noise_kraus_pair: Sequence[
    tuple[GateNoiseInstruction, Sequence[Sequence[Sequence[float]]]]
] = [
    (
        ResetNoise(0.003, 0.002, [0], [names.H]),
        [
            # ResetNoise(p0, p1, ...),
            # sqrt(1 - p0 - p1) * [[1, 0], [0, 1]]
            # [[sqrt(p0), 0], [0, 0]]
            # [[0, sqrt(p0)], [0, 0]]
            # [[0, 0], [sqrt(p1), 0]]
            # [[0, 0], [0, sqrt(p1)]]
            [[0.9974968671630001, 0], [0, 0.9974968671630001]],
            [[0.05477225575051661, 0], [0, 0]],
            [[0, 0.05477225575051661], [0, 0]],
            [[0, 0], [0.044721359549995794, 0]],
            [[0, 0], [0, 0.044721359549995794]],
        ],
    ),
    (
        PhaseDampingNoise(0.003, [0], [names.H]),
        [
            # PhaseDampingNoise(rate, ...),
            # [[1, 0], [0, sqrt(1 - rate)]]
            # [[0, 0], [0, sqrt(rate)]
            [[1, 0], [0, 0.9984988733093293]],
            [[0, 0], [0, 0.05477225575051661]],
        ],
    ),
    (
        AmplitudeDampingNoise(0.003, 0.1, [0], [names.H]),
        [
            # AmplitudeDampingNoise(rate, esp, ...),
            # sqrt(1 - esp) * [[1, 0], [0, sqrt(1 - rate)]]
            # sqrt(1 - esp) * [[0, sqrt(rate)], [0, 0]]
            # sqrt(esp) * [[sqrt(1 - rate), 0], [0, 1]]
            # sqrt(esp) * [[0, 0], [sqrt(rate), 0]]
            [[0.9486832980505138, 0], [0, 0.9472592042308167]],
            [[0, 0.05196152422706632], [0, 0]],
            [[0.31575306807693887, 0], [0, 0.31622776601683794]],
            [[0, 0], [0.017320508075688773, 0]],
        ],
    ),
    (
        PhaseAmplitudeDampingNoise(0.004, 0.003, 0.1, [0], [names.H]),
        [
            # PhaseAmplitudeDampingNoise(prate, arate, esp, ...),
            # sqrt(1 - esp) * [[1, 0], [0, sqrt(1 - prate - arate)]]
            # [[0, sqrt(1 - esp) * sqrt(arate)], [0, 0]]
            # [[0, 0], [0, sqrt(1 - esp) * sqrt(prate)]]
            # sqrt(esp) * [[sqrt(1 - prate - arate), 0], [0, 1]]
            # [[0, 0], [sqrt(esp) * sqrt(arate), 0]]
            # [[sqrt(esp) * sqrt(prate), 0], [0, 0]]
            [[0.9486832980505138, 0], [0, 0.9453570753953238]],
            [[0, 0.05196152422706632], [0, 0]],
            [[0, 0], [0, 0.05999999999999999]],
            [[0.3151190251317746, 0], [0, 0.31622776601683794]],
            [[0, 0], [0.017320508075688773, 0]],
            [[0.02, 0], [0, 0]],
        ],
    ),
    (
        ThermalRelaxationNoise(0.05, 0.04, 50.0, 0.1, [0], [names.H]),
        [
            # Please refer ThermalRelaxationNoise.get_kraus_operator_sequence()
            [[0.9486832980505138, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.3162277660168379, 0.0]],
            [[0.0, 0.9486832980505138], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.3162277660168379]],
        ],
    ),
    (
        KrausNoise(_kraus_list(), [0], [names.H]),
        [
            # Same as _kraus_list()
            [[0.9964938534682489, 0], [0, 0.9964938534682489]],
            [[0.07071067811865475, 0], [0, 0]],
            [[0, 0.07071067811865475], [0, 0]],
            [[0, 0], [0.044721359549995794, 0]],
            [[0, 0], [0, 0.044721359549995794]],
        ],
    ),
]


def test_kraus_operators() -> None:
    for noise, kraus in _noise_kraus_pair:
        print(noise.kraus_operators)
        assert np.allclose(noise.kraus_operators, kraus)


def test_custom_gate_filter() -> None:
    noises = [
        BitFlipNoise(0.004, [], [names.H, names.X]),
        PhaseFlipNoise(0.003, [], []),
        DepolarizingNoise(0.002, [3], []),
    ]

    model = NoiseModel()
    model.add_noise(noises[0])
    model.add_noise(
        noises[1],
        lambda gate: len(gate.target_indices) + len(gate.control_indices) == 2,
    )
    model.add_noise(
        noises[2],
        lambda gate: len(gate.target_indices) == 1 and len(gate.control_indices) == 1,
    )

    assert list(model.noises_for_gate(gates.H(1))) == [((1,), noises[0])]
    assert list(model.noises_for_gate(gates.SWAP(3, 1))) == [
        ((3,), noises[1]),
        ((1,), noises[1]),
    ]
    assert list(model.noises_for_gate(gates.CNOT(0, 3))) == [
        ((0,), noises[1]),
        ((3,), noises[1]),
        ((3,), noises[2]),
    ]
