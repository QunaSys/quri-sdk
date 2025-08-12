# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import copy
from typing import Callable, Sequence, Union, overload

from qulacs import ParametricQuantumCircuit as QulacsParametricQuantumCircuit
from qulacs import QuantumCircuit as QulacsQuantumCircuit

import quri_parts.qulacs.circuit as qlc
from quri_parts.circuit import (
    ImmutableLinearMappedParametricQuantumCircuit,
    ImmutableParametricQuantumCircuit,
    ImmutableQuantumCircuit,
    LinearMappedParametricQuantumCircuit,
    ParametricQuantumCircuit,
    QuantumCircuit,
)


def compile_circuit(circuit: ImmutableQuantumCircuit) -> "_QulacsCircuit":
    """Compiles a quri-parts circuit into an `ImmutableQuantumCircuit` that
    holds the corresponding qulacs circuit on memory. The qulacs circuit can be
    accessed by the `.qulacs_circuit` property.

    Example:

    >>> circuit = QuantumCircuit(2)
    >>> circuit.add_X_gate(0)
    >>> circuit.add_X_gate(1)

    Get compiled circuit

    >>> compiled_circuit = compile_circuit(circuit)
    >>> compiled_qulacs_circuit = compiled_circuit.qulacs_circuit
    """
    return _QulacsCircuit(circuit)


@overload
def compile_parametric_circuit(
    circuit: ParametricQuantumCircuit,
) -> "_QulacsUnboundParametricCircuit":
    ...


@overload
def compile_parametric_circuit(
    circuit: LinearMappedParametricQuantumCircuit,
) -> "_QulacsLinearMappedUnboundParametricCircuit":
    ...


def compile_parametric_circuit(
    circuit: Union[ParametricQuantumCircuit, LinearMappedParametricQuantumCircuit]
) -> Union[
    "_QulacsUnboundParametricCircuit", "_QulacsLinearMappedUnboundParametricCircuit"
]:
    """Compiles a:

        - quri-parts `ParametricQuantumCircuit` into an
            `ImmutableParametricQuantumCircuit` that holds the corresponding
            qulacs parametric circuit on memory.
        - quri-parts `LinearMappedParametricQuantumCircuit` into an
            `ImmutableParametricQuantumCircuit` that holds the corresponding
            qulacs parametric circuit on memory.

    The qulacs circuit can be accessed by the `.qulacs_circuit` property.

    The parameter mapper that maps the quri-parts circuit parameters to qulacs
    parametric circuit parameters can be accessed by the `.param_mapper` property.

    Example:

    >>> parametric_circuit = ParametricQuantumCircuit(2)
    >>> parametric_circuit.add_ParametricRX_gate(0)
    >>> parametric_circuit.add_ParametricRX_gate(1)

    1. Get compiled parametric circuit

    >>> compiled_parametric_circuit = compile_circuit(circuit)
    >>> compiled_qulacs_circuit = compiled_circuit.qulacs_circuit

    2. Get qulacs circuit parameters

    >>> param_mapper = compiled_circuit.param_mapper
    >>> quri_parts_circuit_param = [0, 1]
    >>> qulacs_circuit_param = param_mapper(quri_parts_circuit_param)
    """
    if isinstance(circuit, ParametricQuantumCircuit):
        return _QulacsUnboundParametricCircuit(circuit)
    elif isinstance(circuit, LinearMappedParametricQuantumCircuit):
        return _QulacsLinearMappedUnboundParametricCircuit(circuit)
    else:
        raise ValueError(f"Unsupported parametric circuit type: {type(circuit)}")


# ImmutableQuantumCircuit is not extendable
class _QulacsCircuit(QuantumCircuit):
    _qulacs_circuit: QulacsQuantumCircuit

    def __new__(cls, circuit: ImmutableQuantumCircuit) -> "_QulacsCircuit":
        obj: "_QulacsCircuit" = super().__new__(
            cls, circuit.qubit_count, circuit.cbit_count, circuit.gates
        )  # type: ignore
        obj._qulacs_circuit = qlc.convert_circuit(circuit)
        return obj

    def freeze(self) -> "_QulacsCircuit":
        return self

    @property
    def qulacs_circuit(self) -> QulacsQuantumCircuit:
        return self._qulacs_circuit.copy()


# ImmutableParametricQuantumCircuit is not extendable
class _QulacsUnboundParametricCircuit(ParametricQuantumCircuit):
    _qulacs_circuit: QulacsParametricQuantumCircuit
    _param_mapper: Callable[[Sequence[float]], Sequence[float]]

    def __new__(
        cls, circuit: ImmutableParametricQuantumCircuit
    ) -> "_QulacsUnboundParametricCircuit":
        obj: "_QulacsUnboundParametricCircuit" = super().__new__(
            cls, circuit.qubit_count, circuit.cbit_count
        )  # type: ignore
        obj.extend(circuit)
        obj._qulacs_circuit, obj._param_mapper = qlc.convert_parametric_circuit(circuit)
        return obj

    @property
    def qulacs_circuit(self) -> QulacsParametricQuantumCircuit:
        return self._qulacs_circuit.copy()

    @property
    def param_mapper(self) -> Callable[[Sequence[float]], Sequence[float]]:
        return copy(self._param_mapper)


class _QulacsLinearMappedUnboundParametricCircuit(
    ImmutableLinearMappedParametricQuantumCircuit
):
    def __init__(self, circuit: ImmutableLinearMappedParametricQuantumCircuit):
        super().__init__(circuit)
        (
            self._qulacs_circuit,
            self._qulacs_param_mapper,
        ) = qlc.convert_parametric_circuit(circuit)

    @property
    def qulacs_circuit(self) -> QulacsParametricQuantumCircuit:
        return self._qulacs_circuit.copy()

    @property
    def param_mapper(self) -> Callable[[Sequence[float]], Sequence[float]]:
        return copy(self._qulacs_param_mapper)
