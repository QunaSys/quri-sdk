# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Final, Mapping

SAMPLER_CONTEXTS: Final[Mapping[str, str]] = {
    "create_qulacs_vector_ideal_sampler": "sampler.create_qulacs_vector_ideal_sampler",
    "create_qulacs_vector_sampler": "sampler.create_qulacs_vector_sampler",
    "create_qulacs_vector_concurrent_sampler": "sampler.create_qulacs_vector_concurrent_sampler",
    "create_qulacs_general_vector_sampler": "sampler.create_qulacs_general_vector_sampler",
    "create_qulacs_general_vector_ideal_sampler": "sampler.create_qulacs_general_vector_ideal_sampler",
    "create_qulacs_density_matrix_sampler": "sampler.create_qulacs_density_matrix_sampler",
    "create_qulacs_density_matrix_ideal_sampler": "sampler.create_qulacs_density_matrix_ideal_sampler",
    "create_qulacs_density_matrix_general_sampler": "sampler.create_qulacs_density_matrix_general_sampler",
    "create_qulacs_ideal_density_matrix_general_sampler": "sampler.create_qulacs_ideal_density_matrix_general_sampler",
    "create_qulacs_noisesimulator_sampler": "sampler.create_qulacs_noisesimulator_sampler",
    "create_qulacs_density_matrix_concurrent_sampler": "sampler.create_qulacs_density_matrix_concurrent_sampler",
    "create_qulacs_noisesimulator_concurrent_sampler": "sampler.create_qulacs_noisesimulator_concurrent_sampler",
    "create_qulacs_noisesimulator_general_sampler": "sampler.create_qulacs_noisesimulator_general_sampler",
}

SIMULATOR_CONTEXTS: Final[Mapping[str, str]] = {
    "evaluate_state_to_vector": "simulator.evaluate_state_to_vector",
    "run_circuit": "simulator.run_circuit",
    "create_qulacs_vector_state_sampler": "simulator.create_qulacs_vector_state_sampler",
    "create_concurrent_vector_state_sampler": "simulator.create_concurrent_vector_state_sampler",
    "create_qulacs_ideal_vector_state_sampler": "simulator.create_qulacs_ideal_vector_state_sampler",
    "create_qulacs_density_matrix_state_sampler": "simulator.create_qulacs_density_matrix_state_sampler",
    "create_qulacs_ideal_density_matrix_state_sampler": "simulator.create_qulacs_ideal_density_matrix_state_sampler",
    "create_qulacs_noisesimulator_state_sampler": "simulator.create_qulacs_noisesimulator_state_sampler",
}

NOISESIMULATOR_CONTEXTS: Final = frozenset(
    [
        SAMPLER_CONTEXTS["create_qulacs_noisesimulator_sampler"],
        SAMPLER_CONTEXTS["create_qulacs_noisesimulator_concurrent_sampler"],
        SAMPLER_CONTEXTS["create_qulacs_noisesimulator_general_sampler"],
        SIMULATOR_CONTEXTS["create_qulacs_noisesimulator_state_sampler"],
    ]
)
