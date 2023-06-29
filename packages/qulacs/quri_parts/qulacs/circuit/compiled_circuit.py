from copy import copy
from typing import Callable, Sequence, Union, overload

from qulacs import ParametricQuantumCircuit as QulacsParametricQuantumCircuit
from qulacs import QuantumCircuit as QulacsQuantumCircuit

import quri_parts.qulacs.circuit as qlc
from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    ImmutableQuantumCircuit,
    ImmutableUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuitBase,
    NonParametricQuantumCircuit,
    UnboundParametricQuantumCircuit,
    UnboundParametricQuantumCircuitBase,
)


def compile_circuit(circuit: NonParametricQuantumCircuit) -> "_QulacsCircuit":
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
    circuit: UnboundParametricQuantumCircuit,
) -> "_QulacsUnboundParametricCircuit":
    ...


@overload
def compile_parametric_circuit(
    circuit: LinearMappedUnboundParametricQuantumCircuit,
) -> "_QulacsLinearMappedUnboundParametricCircuit":
    ...


def compile_parametric_circuit(
    circuit: Union[
        UnboundParametricQuantumCircuit, LinearMappedUnboundParametricQuantumCircuit
    ]
) -> Union[
    "_QulacsUnboundParametricCircuit", "_QulacsLinearMappedUnboundParametricCircuit"
]:
    """Compiles a:

        - quri-parts `UnboundParametricQuantumCircuit` into an
            `ImmutableUnboundParametricQuantumCircuit` that holds the corresponding
            qulacs parametric circuit on memory.
        - quri-parts `LinearMappedUnboundParametricQuantumCircuit` into an
            `ImmutableUnboundParametricQuantumCircuit` that holds the corresponding
            qulacs parametric circuit on memory.

    The qulacs circuit can be accessed by the `.qulacs_circuit` property.

    The parameter mapper that maps the quri-parts circuit parameters to qulacs
    parametric circuit parameters can be accessed by the `.param_mapper` property.

    Example:

    >>> parametric_circuit = UnboundParametricQuantumCircuit(2)
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
    if isinstance(circuit, UnboundParametricQuantumCircuit):
        return _QulacsUnboundParametricCircuit(circuit)
    elif isinstance(circuit, LinearMappedUnboundParametricQuantumCircuit):
        return _QulacsLinearMappedUnboundParametricCircuit(circuit)
    else:
        raise ValueError(f"Unsupported parametric circuit type: {type(circuit)}")


class _QulacsCircuit(ImmutableQuantumCircuit):
    def __init__(self, circuit: NonParametricQuantumCircuit):
        super().__init__(circuit)
        self._qulacs_circuit = qlc.convert_circuit(circuit)

    @property
    def qulacs_circuit(self) -> QulacsQuantumCircuit:
        return self._qulacs_circuit.copy()


class _QulacsUnboundParametricCircuit(ImmutableUnboundParametricQuantumCircuit):
    def __init__(self, circuit: UnboundParametricQuantumCircuitBase):
        super().__init__(circuit)
        self._qulacs_circuit, self._param_mapper = qlc.convert_parametric_circuit(
            circuit
        )

    @property
    def qulacs_circuit(self) -> QulacsParametricQuantumCircuit:
        return self._qulacs_circuit.copy()

    @property
    def param_mapper(self) -> Callable[[Sequence[float]], Sequence[float]]:
        return copy(self._param_mapper)


class _QulacsLinearMappedUnboundParametricCircuit(
    ImmutableLinearMappedUnboundParametricQuantumCircuit
):
    def __init__(self, circuit: LinearMappedUnboundParametricQuantumCircuitBase):
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
