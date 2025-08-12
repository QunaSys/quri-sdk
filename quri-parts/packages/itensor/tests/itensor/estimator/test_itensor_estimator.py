# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any, Union

import numpy as np
import pytest

from quri_parts.circuit import (
    ParametricQuantumCircuit,
    ParametricQuantumCircuitProtocol,
)
from quri_parts.core.operator import Operator, PauliLabel, pauli_label
from quri_parts.core.state import ComputationalBasisState, ParametricCircuitQuantumState
from quri_parts.itensor.estimator import (
    create_itensor_mps_concurrent_estimator,
    create_itensor_mps_estimator,
    create_itensor_mps_parametric_estimator,
)


class TestITensorEstimator:
    def test_estimate_pauli_label(self) -> None:
        pauli = pauli_label("Z0 Z2 Z5")
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_itensor_mps_estimator()
        estimate = estimator(pauli, state)
        assert estimate.value == -1
        assert estimate.error == 0.0

        estimator_with_kwargs = create_itensor_mps_estimator(maxdim=2, cutoff=0.01)
        estimate_with_kwargs = estimator_with_kwargs(pauli, state)
        assert estimate_with_kwargs.value == -1
        assert np.isnan(estimate_with_kwargs.error)

    def test_estimate_operator(self) -> None:
        operator = Operator(
            {
                pauli_label("Z0 Z2 Z5"): 0.25,
                pauli_label("Z1 Z2 Z4"): 0.5j,
            }
        )
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_itensor_mps_estimator()
        estimate = estimator(operator, state)
        assert estimate.value == -0.25 + 0.5j
        assert estimate.error == 0.0
        # Note that the results of numerical calculations differ
        # depending on the presence or absence of keyword arguments.
        estimator_with_kwargs = create_itensor_mps_estimator(maxdim=2, cutoff=0.01)
        estimate_with_kwargs = estimator_with_kwargs(operator, state)
        assert estimate_with_kwargs.value == -0.25 + 0.5j
        assert np.isnan(estimate_with_kwargs.error)


class TestITensorConcurrentEstimator:
    def test_invalid_arguments(self) -> None:
        pauli = pauli_label("Z0 Z2 Z5")
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_itensor_mps_concurrent_estimator()
        with pytest.raises(ValueError):
            estimator([], [state])
        with pytest.raises(ValueError):
            estimator([pauli], [])
        with pytest.raises(ValueError):
            estimator([pauli] * 3, [state] * 2)

    @pytest.mark.skip(reason="This test is too slow.")
    @pytest.mark.parametrize(
        "jl_apply_kwargs",
        [
            {},
            {"maxdim": 2, "cutoff": 0.01},
        ],
    )
    def test_concurrent_estimate(self, jl_apply_kwargs: Any) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            pauli_label("Z0 Z2 Z5"),
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            ComputationalBasisState(6, bits=0b110000),
            ComputationalBasisState(6, bits=0b110010),
        ]

        with ProcessPoolExecutor(
            max_workers=2, mp_context=get_context("spawn")
        ) as executor:
            estimator = create_itensor_mps_concurrent_estimator(
                executor, concurrency=2, **jl_apply_kwargs
            )
            result = list(estimator(operators, states))
        assert result[0].value == -1
        assert result[1].value == -0.25 + 0.5j
        if "maxdim" in jl_apply_kwargs or "cutoff" in jl_apply_kwargs:
            assert result[0].error == np.nan
            assert result[1].error == np.nan

    @pytest.mark.skip(reason="This test is too slow.")
    def test_concurrent_estimate_single_state(self) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            pauli_label("Z0 Z2 Z5"),
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            ComputationalBasisState(6, bits=0b110010),
        ]
        with ProcessPoolExecutor(
            max_workers=2, mp_context=get_context("spawn")
        ) as executor:
            estimator = create_itensor_mps_concurrent_estimator(executor, concurrency=2)
            result = list(estimator(operators, states))
        assert result[0].value == -1
        assert result[1].value == -0.25 + 0.5j
        assert result[0].error == 0.0
        assert result[1].error == 0.0

    @pytest.mark.skip(reason="This test is too slow.")
    def test_concurrent_estimate_single_operator(self) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            ComputationalBasisState(6, bits=0b110010),
            ComputationalBasisState(6, bits=0b110011),
        ]
        with ProcessPoolExecutor(
            max_workers=2, mp_context=get_context("spawn")
        ) as executor:
            estimator = create_itensor_mps_concurrent_estimator(executor, concurrency=2)
            result = list(estimator(operators, states))
        assert result[0].value == -0.25 + 0.5j
        assert result[1].value == 0.25 + 0.5j
        assert result[0].error == 0.0
        assert result[1].error == 0.0


def parametric_circuit() -> ParametricQuantumCircuitProtocol:
    circuit = ParametricQuantumCircuit(6)
    circuit.add_RX_gate(0, -math.pi / 4)
    circuit.add_ParametricRX_gate(0)

    circuit.add_RY_gate(2, -math.pi / 4)
    circuit.add_ParametricRY_gate(2)

    circuit.add_H_gate(5)
    circuit.add_RZ_gate(5, -math.pi / 4)
    circuit.add_ParametricRZ_gate(5)

    return circuit


class TestITensorParametricEstimator:
    def test_estimate_pauli_label(self) -> None:
        pauli = pauli_label("Y0 X2 Y5")
        state = ParametricCircuitQuantumState(6, parametric_circuit())
        estimator = create_itensor_mps_parametric_estimator()
        expected_list = [
            0.35355339059327373 + 0j,
            0.5 + 0j,
            0.5 + 0j,
            0.5 + 0j,
        ]
        for i, params in enumerate(
            [
                [0.0, 0.0, 0.0],
                [-math.pi / 4, 0.0, 0.0],
                [0.0, -math.pi / 4, 0.0],
                [0.0, 0.0, -math.pi / 4],
            ]
        ):
            estimate = estimator(pauli, state, params)
            assert estimate.value == pytest.approx(expected_list[i])
            assert estimate.error == 0.0

    def test_estimate_operator(self) -> None:
        operator = Operator(
            {
                pauli_label("Y0 X2 Y5"): 0.25,
                pauli_label("Z1 X2 Z4"): 0.5j,
            }
        )
        state = ParametricCircuitQuantumState(6, parametric_circuit())
        estimator = create_itensor_mps_parametric_estimator()
        expected_list = [
            0.08838834764831843 - 0.35355339059327373j,
            0.12499999999999997 - 0.3535533905932736j,
            0.125 - 0.5j,
            0.125 - 0.35355339059327373j,
        ]
        for i, params in enumerate(
            [
                [0.0, 0.0, 0.0],
                [-math.pi / 4, 0.0, 0.0],
                [0.0, -math.pi / 4, 0.0],
                [0.0, 0.0, -math.pi / 4],
            ]
        ):
            estimate = estimator(operator, state, params)
            assert estimate.value == pytest.approx(expected_list[i])
            assert estimate.error == 0.0

    def test_estimate_operator_with_jl_apply_kwargs(self) -> None:
        operator = Operator(
            {
                pauli_label("Y0 X2 Y5"): 0.25,
                pauli_label("Z1 X2 Z4"): 0.5j,
            }
        )

        circuit = ParametricQuantumCircuit(8)
        for _ in range(2):
            for i in range(4):
                circuit.add_X_gate(i)
            for i in range(8):
                circuit.add_ParametricRY_gate(i)
                circuit.add_ParametricRZ_gate(i)
            for i in [1, 3, 5]:
                circuit.add_CZ_gate(i, i + 1)
            for i in [0, 2, 4, 6]:
                circuit.add_CZ_gate(i, i + 1)
            for i in range(8):
                circuit.add_ParametricRY_gate(i)
                circuit.add_ParametricRZ_gate(i)

        state = ParametricCircuitQuantumState(8, circuit)
        params = [i / 64 for i in range(64)]
        estimator = create_itensor_mps_parametric_estimator()
        estimate = estimator(operator, state, params)
        assert estimate.value == pytest.approx(
            0.024541701187281842 + 0.02441865385682042j
        )
        assert estimate.error == 0
        # Note that the results of numerical calculations differ
        # depending on the presence or absence of keyword arguments.
        estimator_with_kwargs = create_itensor_mps_parametric_estimator(
            maxdim=2, cutoff=0.01
        )
        estimate_with_kwargs = estimator_with_kwargs(operator, state, params)
        assert estimate_with_kwargs.value == pytest.approx(
            0.02485533900849108 + 0.029037213118795355j
        )
        assert np.isnan(estimate_with_kwargs.error)
