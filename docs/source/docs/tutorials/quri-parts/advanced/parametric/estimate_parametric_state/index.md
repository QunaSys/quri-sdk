# Parametric States and Parametric Estimators

This tutorial aims to introduce parametric states and parametric estimators, which are also important building blocks of variational algorithms.

## Parametric state

In QURI Parts, there are 2 types of parametric states:

- `ParametricCircuitQuantumState`: A state with parametric circuit acting on a zero state.
- `ParametricQuantumStateVector`: A state with parametric circuit acting on a fixed state vector.

Here, parametric circuit can be either an `UnboundParametricQuantumCircuit` or a `LinearMappedUnboundParametricQuantumCircuit`. The parametric state can be created with the `quantum_state` function. Let's first create a parametric circuit.


```python
import numpy as np
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit, CONST

linear_param_circuit = LinearMappedUnboundParametricQuantumCircuit(2)
theta, phi = linear_param_circuit.add_parameters("theta", "phi")

linear_param_circuit.add_H_gate(0)
linear_param_circuit.add_CNOT_gate(0, 1)
linear_param_circuit.add_ParametricRX_gate(0, {theta: 1/2, phi: 1/3, CONST: np.pi/2})
linear_param_circuit.add_ParametricRY_gate(0, {theta: -1/2, phi: 1/3})
linear_param_circuit.add_ParametricRZ_gate(1, {theta: 1/3, phi: -1/2, CONST: -np.pi/2})
```

Now, we create parametric state with it.


```python
from quri_parts.core.state import quantum_state

# ParametricCircuitQuantumState
param_circuit_state = quantum_state(n_qubits=2, circuit=linear_param_circuit)
print(param_circuit_state)

# ParametricQuantumStateVector
param_state_vector = quantum_state(n_qubits=2, circuit=linear_param_circuit, vector=np.array([0.5, -0.5, 0.5j, -0.5j]))
print(param_state_vector)
```
```output
ParametricCircuitQuantumState(n_qubits=2, circuit=<quri_parts.circuit.circuit_linear_mapped.ImmutableLinearMappedParametricQuantumCircuit object at 0x7f3e0137b6d0>)
ParametricQuantumStateVector(n_qubits=2, circuit=<quri_parts.circuit.circuit_linear_mapped.ImmutableLinearMappedParametricQuantumCircuit object at 0x7f3e50044a50>, vector=[ 0.5+0.j  -0.5+0.j   0. +0.5j -0. -0.5j])
```
We can assign concrete values to the circuit inside the parametric states with the `.bind_parameters` state. After binding the parameters, `ParametricCircuitQuantumState` becomes a `GeneralCircuitQuantumState` and `ParametricQuantumStateVector` becomes a `QuantumStateVector`.


```python
params = [0.1, 0.2]

# ParametricCircuitQuantumState -> GeneralCircuitQuantumState
circuit_state = param_circuit_state.bind_parameters(params)

# ParametricQuantumStateVector -> QuantumStateVector
vector_state = param_state_vector.bind_parameters(params)
```

## Parametric estimators

In addition to the `QuantumEstimator`s and `ConcurrentQuantumEstimator`s introduced in the [estimator tutorial](../../../basics/estimators/index.md), QURI Parts also provide estimators for parametric states. They are:

- `ParametricQuantumEstimator`: Estimate an operator's expectation value for a parametric state with a set of parameters.
- `ConcurrentParametricQuantumEstimator`: Estimate an operator's expectation value for a parametric state with multiple sets of parameters.

Note that different from `ConcurrentQuantumEstimator`, a `ConcurrentParametricQuantumEstimator` only performs estimation for a parametric state at a time. The concurrent estimation is done across multiple sets of parameters.

### Parametric estimate with a `QuantumEstimator`

A `GeneralCircuitQuantumState` can be created from a circuit obtained by binding values to a parametric circuit and a `QuantumEstimator` can be used to estimate an expectation value of an operator for the state:


```python
from quri_parts.core.state import quantum_state
from quri_parts.qulacs.estimator import create_qulacs_vector_estimator

circuit = linear_param_circuit.bind_parameters([0.2, 0.3])
circuit_state = quantum_state(2, circuit=circuit)

from quri_parts.core.operator import Operator, pauli_label
op = Operator({
    pauli_label("X0 Y1"): 0.5 + 0.5j,
    pauli_label("Z0 X1"): 0.2,
})


estimator = create_qulacs_vector_estimator()

estimate = estimator(op, circuit_state)
print(f"Estimated expectation value: {estimate.value}")
```
```output
Estimated expectation value: (-0.6935980009039755-0.49826489335027957j)
```
However, it is more straightforward to use `ParametricCircuitQuantumState`. It has a few benefits:

- It makes it clear that the state is parametric and possible to treat parameter-related problems in terms of quantum state (e.g. gradient of an expectation value for the state with respect to its parameters).

- It may improve performance for some circuit simulators (e.g. Qulacs).

With this in mind, we introduce solid implementations of these parametric estimators provided by the `quri_parts.qulacs` package.

### The Qulacs parametric estimators

The `quri_parts.qulacs` package provides both vector parametric estimators and density matrix parametric estimators. Here we demonstrate the Qulacs vector parametric estimators for performing noiseless estimation for parametric states. They are generated by the 

- `create_qulacs_vector_parametric_estimator`
- `create_qulacs_vector_concurrent_parametric_estimator`

functions.


```python
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_parametric_estimator,
    create_qulacs_vector_concurrent_parametric_estimator,
)

parametric_estimator = create_qulacs_vector_parametric_estimator()
concurrent_parametric_estimator = create_qulacs_vector_concurrent_parametric_estimator()
```

Next, let's create an operator for us to estimate.


```python
from quri_parts.core.operator import Operator, pauli_label

operator = Operator(
    {
        pauli_label("X0 Y1"): 0.5 + 0.5j,
        pauli_label("Z0 X1"): 0.2,
    }
)
```

### Estimate for parametric circuit state

The Qulacs parametric estimator can be used to estimate the expectation values for parametric circuit state.


```python
# Estimate for parametric circuit state
print("Estimate with a set of circuit parameter:")
print(parametric_estimator(operator, param_circuit_state, [0.1, 0.2]))

# Concurrent estimate for parametric circuit state
print("")
print("Concurrent Estimate with two sets of circuit parameters:")
print(concurrent_parametric_estimator(operator, param_circuit_state, [[0.1, 0.2], [0.3, 0.4]]))
```
```output
Estimate with a set of circuit parameter:
_Estimate(value=(-0.6962182648863564-0.4982686669563893j), error=0.0)

Concurrent Estimate with two sets of circuit parameters:
[_Estimate(value=(-0.6962182648863564-0.4982686669563893j), error=0.0), _Estimate(value=(-0.6896044045783039-0.49823172433086493j), error=0.0)]
```
### Estimate for parametric vector state

The Qulacs parametric estimator can also be used to estimate the expectation values for parametric state vectors.


```python
# Estimate for parametric state vector
print("Estimate with a set of circuit parameter:")
print(parametric_estimator(operator, param_state_vector, [0.1, 0.2]))

# Concurrent estimate for parametric state vector
print("")
print("Concurrent Estimate with two sets of circuit parameters:")
print(concurrent_parametric_estimator(operator, param_state_vector, [[0.1, 0.2], [0.3, 0.4]]))
```
```output
Estimate with a set of circuit parameter:
_Estimate(value=(-0.02316087896430696+6.461699812418398e-05j), error=0.0)

Concurrent Estimate with two sets of circuit parameters:
[_Estimate(value=(-0.02316087896430696+6.461699812418398e-05j), error=0.0), _Estimate(value=(-0.055857039676488814-0.00023256584020162063j), error=0.0)]
```