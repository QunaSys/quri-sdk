# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional

import numpy as np
from quri_parts.core.state import CircuitQuantumState

from quri_algo.problem.operators.hamiltonian import QubitHamiltonian
from quri_algo.state.adiabatic_state_preparation.interface import (
    AdiabaticTimeEvolutionStateFactoryBase,
    HamiltonianMapping,
    TECircuitFactoryConstructor,
)


def get_linear_hamiltonian_mapping(
    h_0: QubitHamiltonian, h_1: QubitHamiltonian, evolution_time: float
) -> HamiltonianMapping:
    if not h_0.n_qubit == h_1.n_qubit:
        raise ValueError(
            f"Provided QubitHamiltonian qubit counts, {h_0.n_qubit} and {h_1.n_qubit}, do not match"
        )

    def hamiltonian_mapping(t: float) -> QubitHamiltonian:
        ratio = t / evolution_time
        h = h_0.qubit_hamiltonian * (1 - ratio) + h_1.qubit_hamiltonian * ratio
        return QubitHamiltonian(h_0.n_qubit, h)

    return hamiltonian_mapping


class AdiabaticTimeEvolutionStateFactory(AdiabaticTimeEvolutionStateFactoryBase):
    """Base class for state preparation that relies on Hamiltonian
    simulation."""

    def __init__(
        self,
        hamiltonian_mapping: HamiltonianMapping,
        TECircuitFactory: TECircuitFactoryConstructor,
    ) -> None:
        self.hamiltonian_mapping = hamiltonian_mapping
        self.TECircuitFactory = TECircuitFactory

    def verify_inputs(
        self, discretization: int, interp_function: Callable[[float], float]
    ) -> None:
        if discretization < 2:
            raise ValueError("Discretization should be greater than 1")
        if interp_function(0.0) != 0.0 or interp_function(1.0) != 1.0:
            raise ValueError(
                "Interpolation function should map the range (0;1) to (0;1)"
            )

    def __call__(
        self,
        evolution_time: float,
        discretization: int,
        initial_state: CircuitQuantumState,
        interp_function: Callable[[float], float] = lambda x: x,
        stop_at_time: Optional[float] = None,
        stop_at_iteration: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> CircuitQuantumState:
        r"""Perform adiabatic time-evolution on the provided state.

        Perform time-evolution on the provided :class:`HamiltonianMapping`. Time-evolution intervals are inferred by the interpolation function. Variable and keyword arguments are passed to the time-evolution circuit factory constructor.

        Arguments:
            evolution_time: total time-evolution
            discretization: number of times the time-evolution circuit factory is called
            initial_state: initial state to perform time-evolution on, if provided
            interp_function: monotonically increasing function, which maps the range (0;1) to (0;1). Is applied to the time-interval
            stop_at_time: stops the time-evolution at the provided time to obtain an intermediate state
            stop_at_iteration: stops the time-evolution at the provided iteration to obtain an intermediate state
        """
        self.verify_inputs(discretization, interp_function)

        assert (stop_at_time is None) or (
            stop_at_iteration is None
        ), "Only one of stop_at_time or stop_at_iteration may be provided"

        times = [
            interp_function(t / evolution_time) * evolution_time
            for t in np.linspace(0.0, evolution_time, discretization)
        ]

        for i, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
            if stop_at_iteration is not None and stop_at_iteration <= i:
                break
            if stop_at_time is not None and stop_at_time <= t0:
                break
            s = (t1 + t0) / 2
            dt = t1 - t0
            h = self.hamiltonian_mapping(s)
            te_factory = self.TECircuitFactory(h, *args, **kwargs)
            initial_state = initial_state.with_gates_applied(te_factory(dt))

        return initial_state
