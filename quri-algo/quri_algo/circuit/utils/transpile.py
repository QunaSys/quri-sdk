# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps
from typing import TYPE_CHECKING, Callable, cast

from quri_parts.circuit import NonParametricQuantumCircuit
from typing_extensions import Concatenate, ParamSpec

if TYPE_CHECKING:
    from quri_algo.circuit.interface import CircuitFactory


P = ParamSpec("P")


def apply_transpiler(
    circuit_factory_call: Callable[
        Concatenate["CircuitFactory", P], NonParametricQuantumCircuit
    ],
) -> Callable[Concatenate["CircuitFactory", P], NonParametricQuantumCircuit]:
    """A decorator that applies the transpiler to the circuit from the circuit
    factory.

    It has to be applied to the __call__ method in ALL subclasses of
    `CircuitFactory`s.
    """

    @wraps(circuit_factory_call)
    def f(
        circuit_factory: "CircuitFactory", *args: P.args, **kwargs: P.kwargs
    ) -> NonParametricQuantumCircuit:
        circuit = circuit_factory_call(circuit_factory, *args, **kwargs)
        if circuit_factory.transpiler is None:
            return circuit
        return circuit_factory.transpiler(circuit)

    return cast(
        Callable[
            Concatenate["CircuitFactory", P],
            NonParametricQuantumCircuit,
        ],
        f,
    )
