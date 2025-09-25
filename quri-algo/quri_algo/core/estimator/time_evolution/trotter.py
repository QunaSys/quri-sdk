# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Union

from quri_parts.circuit.transpile import CircuitTranspiler
from quri_parts.core.sampling import Sampler, StateSampler

from quri_algo.circuit.time_evolution.trotter_time_evo import (
    TrotterControlledTimeEvolutionCircuitFactory,
)
from quri_algo.core.estimator import StateT
from quri_algo.core.estimator.interface import State
from quri_algo.problem import QubitHamiltonian

from .interface import TimeEvolutionHadamardTest


class TrotterTimeEvolutionHadamardTest(TimeEvolutionHadamardTest[StateT]):
    r"""Performs :math:`\langle e^{-iHt} \rangle` base on Trotterized
    implementation of the time evolution operator :math:`e^{-iHt}`."""

    def __init__(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        sampler: Union[Sampler, StateSampler[State]],
        *,
        time_step: float | None = None,
        n_trotter: int | None = None,
        trotter_order: int = 1,
        transpiler: CircuitTranspiler | None = None
    ):
        self.time_step = time_step
        self.n_trotter = n_trotter
        self.trotter_order = trotter_order
        controlled_time_evolution_factory = (
            TrotterControlledTimeEvolutionCircuitFactory(
                qubit_hamiltonian,
                time_step=time_step,
                n_trotter=n_trotter,
                trotter_order=trotter_order,
                transpiler=transpiler,
            )
        )
        super().__init__(
            qubit_hamiltonian,
            controlled_time_evolution_factory,
            sampler,
            transpiler=transpiler,
        )
