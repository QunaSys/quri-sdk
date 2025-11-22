# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock

import numpy as np
from quri_parts.algo.optimizer import OptimizerStatus
from quri_parts.circuit import (
    ImmutableBoundParametricQuantumCircuit,
    ImmutableQuantumCircuit,
    ParametricQuantumCircuit,
)

from quri_algo.algo.compiler.qaqc import QAQC
from quri_algo.algo.interface import Analyzer, LoweringLevel
from quri_algo.algo.utils.variational_solvers import QURIPartsVariationalSolver
from quri_algo.circuit.interface import CircuitFactory
from quri_algo.core.cost_functions.base_classes import LocalCostFunction


def test_qaqc() -> None:
    cost_fn = Mock(spec=LocalCostFunction)
    optimizer = Mock()
    ansatz = Mock(spec=ParametricQuantumCircuit)
    analyze_result = Mock()
    analyze_result.lowering_level = LoweringLevel.ArchInstruction
    analyze_result.qubit_count = 4
    analyze_result.latency = Mock()
    analyze_result.latency.in_ns = Mock(return_value=1000)
    vm = Mock()
    vm.analyze = Mock(return_value=analyze_result, spec=Analyzer)
    circuit_factory = Mock(spec=CircuitFactory)
    circuit_factory.qubit_count = 4
    bound_circuit = Mock(spec=ImmutableBoundParametricQuantumCircuit)
    bound_circuit.qubit_count = 4
    bound_circuit.gates = []
    returned_circuit = Mock(spec=ImmutableQuantumCircuit)
    returned_circuit.qubit_count = 4
    returned_circuit.gates = []
    ansatz.bind_parameters = Mock(return_value=bound_circuit)
    ansatz.param_mapping = Mock()
    ansatz.param_mapping.out_params = Mock()
    ansatz.param_mapping.out_params.__len__ = Mock(return_value=32)
    circuit_factory.return_value = returned_circuit

    evolution_time = 0.5
    opt_params = np.array([0.5, 0.1, 0.3])
    ansatz.parameter_count = 16
    optimizer_state_success = Mock()
    optimizer_state_success.status = OptimizerStatus.SUCCESS
    optimizer_state_converged = Mock()
    optimizer_state_converged.status = OptimizerStatus.CONVERGED
    optimizer_state_converged.params = opt_params
    optimizer_state_converged.gradcalls = 16
    optimizer_state_converged.funcalls = 4
    result = Mock()
    result.optimizer_history = [optimizer_state_converged]
    optimizer.get_init_state = Mock(return_value=optimizer_state_success)
    optimizer.step = Mock(return_value=optimizer_state_converged)
    solver = QURIPartsVariationalSolver(optimizer)

    qaqc = QAQC(cost_fn, solver)
    qaqc.run(circuit_factory, ansatz, evolution_time=evolution_time)

    assert optimizer.get_init_state.call_count == 1
    assert optimizer.step.call_count == 1
    ansatz.bind_parameters.assert_called_with(opt_params.tolist())

    analysis = qaqc.analyze(
        circuit_factory,
        ansatz,
        result,
        vm.analyze,
        evolution_time=evolution_time,
        circuit_execution_multiplier=500,
    )

    assert analysis.lowering_level == analyze_result.lowering_level
    assert list(analysis.circuit_latency.values())[0] == analyze_result.latency
    assert list(analysis.circuit_qubit_count.values())[0] == analyze_result.qubit_count
    assert analysis.total_latency.in_ns() == 500 * 1000 * (32 * 16 * 2 + 4)
    assert analysis.max_physical_qubit_count == 4

    analyze_result.latency.in_ns.assert_called_once()
