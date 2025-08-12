# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Final, Optional, Sequence

import numpy as np
from numpy.random import default_rng
from quri_parts.algo.optimizer import CostFunction as QPCostFunction
from quri_parts.algo.optimizer import GradientFunction, OptimizerState, Params
from quri_parts.circuit import (
    ImmutableQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
    NonParametricQuantumCircuit,
)
from quri_parts.circuit.parameter_shift import ShiftedParameters

from quri_algo.algo.compiler.base_classes import (
    CompilationAnalysis,
    CompilationResult,
    QuantumCompiler,
)
from quri_algo.algo.interface import Analyzer
from quri_algo.algo.utils.timer import timer
from quri_algo.algo.utils.variational_solvers import (
    VariationalSolver,
    create_default_variational_solver,
)
from quri_algo.circuit.interface import CircuitFactory
from quri_algo.core.cost_functions.base_classes import CostFunction
from quri_algo.core.cost_functions.hilbert_schmidt_test import (
    create_default_hilbert_schmidt_test,
)
from quri_algo.core.cost_functions.utils import prepare_circuit_hilbert_schmidt_test


class QAQC(QuantumCompiler):
    """Quantum-Assisted Quantum Compilation (QAQC)

    This class provides an implementation of QAQC https://quantum-
    journal.org/papers/q-2019-05-13-140/. The optimize method returns a
    compiled circuit along with the converged state of the optimizer. It
    is also possible to use the __call__ method to return the compiled
    circuit only.
    """

    _name: Final[str] = "Quantum-Assisted Quantum Compilation (QAQC)"

    @property
    def name(self) -> str:
        return self._name

    def __init__(
        self,
        cost_fn: CostFunction = create_default_hilbert_schmidt_test(),
        solver: VariationalSolver = create_default_variational_solver(),
    ):
        self._cost_fn = cost_fn
        self._solver = solver

    @property
    def cost_function(self) -> CostFunction:
        return self._cost_fn

    @property
    def solver(self) -> VariationalSolver:
        return self._solver

    def get_cost_fn(
        self,
        ansatz: LinearMappedUnboundParametricQuantumCircuit,
        target_circuit: ImmutableQuantumCircuit,
    ) -> QPCostFunction:
        def cost(params: Params) -> float:
            trial_circuit = ansatz.bind_parameters(list(params.tolist()))
            return self.cost_function(target_circuit, trial_circuit).value.real

        return cost

    def get_gradient_fn(
        self,
        ansatz: LinearMappedUnboundParametricQuantumCircuit,
        target_circuit: ImmutableQuantumCircuit,
    ) -> GradientFunction:
        def gradient(params: Params) -> Params:
            """Parameter shift gradient function.

            This calculates the gradient based on the parameter shift rule, as in (https://quri-parts.qunasys.com/docs/tutorials/advanced/parametric/gradient_estimators/#parameter-shift-gradient-estimator)
            """

            param_mapping = ansatz.param_mapping
            raw_circuit = ansatz.primitive_circuit()
            parameter_shift = ShiftedParameters(param_mapping)
            derivatives = parameter_shift.get_derivatives()
            shifted_parameters_and_coefs = [
                d.get_shifted_parameters_and_coef(list(params)) for d in derivatives
            ]

            gate_params = set()
            for params_and_coefs in shifted_parameters_and_coefs:
                for p, _ in params_and_coefs:
                    gate_params.add(p)
            gate_params_list = list(gate_params)

            estimates = [
                self.cost_function(target_circuit, raw_circuit.bind_parameters(list(p)))
                for p in gate_params_list
            ]
            estimates_dict = dict(zip(gate_params_list, estimates))

            gradient = []
            for params_and_coefs in shifted_parameters_and_coefs:
                g = 0.0
                for p, c in params_and_coefs:
                    g += estimates_dict[p].value.real * c.real
                gradient.append(g)

            return np.array(gradient)

        return gradient

    def _optimize(
        self,
        circuit_factory: CircuitFactory,
        ansatz: LinearMappedUnboundParametricQuantumCircuit,
        init_params: Optional[Params] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[NonParametricQuantumCircuit, Sequence[OptimizerState]]:
        if not isinstance(circuit_factory, CircuitFactory):
            raise TypeError(
                "Target circuit factory should be an instance of CircuitFactory."
            )
        target_circuit = circuit_factory(*args, **kwargs)

        cost = self.get_cost_fn(ansatz, target_circuit)
        gradient = self.get_gradient_fn(ansatz, target_circuit)

        if init_params is None:
            rng = default_rng()
            init_params = rng.random(ansatz.parameter_count) * 2 * np.pi

        optimizer_history = self.solver(cost, gradient, init_params)

        opt_params = optimizer_history[-1].params
        opt_circuit = ansatz.bind_parameters(list(opt_params.tolist()))

        return opt_circuit, optimizer_history

    @timer
    def run(
        self,
        circuit_factory: CircuitFactory,
        ansatz: LinearMappedUnboundParametricQuantumCircuit,
        init_params: Optional[Params] = None,
        *args: Any,
        **kwargs: Any,
    ) -> CompilationResult:
        """This function initiates the variational optimization.

        Arguments:
        circuit_factory - Circuit factory which generates the target circuit for the compilation
        ansatz - Parametric ansatz circuit that is used to approximate the target circuit
        init_params - initial variational parameters for the optimization

        variable arguments if any are passed to the circuit_factory's __call__ method

        Return value:
        Tuple of compiled circuit and the state of the optimizer
        """
        optimized_circuit, optimizer_history = self._optimize(
            circuit_factory, ansatz, init_params, *args, **kwargs
        )

        return CompilationResult(
            algorithm=self,
            optimizer_history=optimizer_history,
            optimized_circuit=optimized_circuit,
        )

    def analyze(
        self,
        circuit_factory: CircuitFactory,
        ansatz: LinearMappedUnboundParametricQuantumCircuit,
        result: CompilationResult,
        analyzer: Analyzer,
        circuit_execution_multiplier: Optional[int] = None,
        concurrency: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> CompilationAnalysis:
        """This function analyses the quantum resources required to run the
        quantum compilation.

        Arguments:
        circuit_factory - Circuit factory which generates the target circuit for the compilation
        ansatz - Parametric ansatz circuit that is used to approximate the target circuit
        optimizer_state - state of the optimizer after the optimization
        circuit_execution_multiplier - multiplier for the number of circuit executions, e.g. number of samples for each estimate

        variable arguments if any are passed to the circuit_factory's __call__ method

        Return value:
        Tuple of compiled circuit and the state of the optimizer
        """

        circuit_list = []
        optimizer_history = result.optimizer_history
        for optimizer_state in optimizer_history:
            target_circuit = circuit_factory(*args, **kwargs)
            combined_circuit = prepare_circuit_hilbert_schmidt_test(
                target_circuit,
                ansatz.bind_parameters(list(optimizer_state.params.tolist())),
            ).circuit
            circuit_list.append(combined_circuit)
        total_circuit_executions = (
            len(ansatz.param_mapping.out_params) * 2 * optimizer_state.gradcalls
            + optimizer_state.funcalls
        )
        if circuit_execution_multiplier is not None:
            total_circuit_executions *= circuit_execution_multiplier
        circuit_analysis = analyzer(
            circuit_list[-1]
        )  # QAQC analysis can be assumed to result identically for each circuit
        analysis = CompilationAnalysis.from_circuit_list(
            circuit_analysis, total_circuit_executions, circuit_list, concurrency
        )

        return analysis
