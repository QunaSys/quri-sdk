# Simulators

Sometimes when we assemble algorithms, it is useful to check whether or not the resulting state is consistent with our expectation. As the state is not something we have access to with experiment, we need to use a simulator to obtain the exact final output state. In QURI Parts, a simulator is any function that work with the explicit state vector or density matrices.

## Qulacs simulators

In the `quri_parts.qulacs.simulator` module, we provide 3 simulator features for state vectors.

- `evaluate_state_to_vector`
- `run_circuit`
- `get_marginal_probability`

We introduce their functionalities in the following sections.

### `evaluate_state_to_vector`

`evaluate_state_to_vector` is a function that converts any non-parametric states into a `QuantumStateVector` with an empty circuit.


```python
from quri_parts.qulacs.simulator import evaluate_state_to_vector
from quri_parts.core.state import quantum_state
```

#### Converts a `ComputationalBasisState`
For example, `bits = 2 = 0b10` represents $|10\rangle$. Note that vectors are arranged as follows: |00>, |01>, |10>, |11>, ... .


```python
comp_state = quantum_state(n_qubits=2, bits=2)

out_state = evaluate_state_to_vector(comp_state)

print("State vector:")
print(out_state.vector)
print("")
print("Circuit:")
print(out_state.circuit.gates)
```
```output
State vector:
[0.+0.j 0.+0.j 1.+0.j 0.+0.j]

Circuit:
()
```
#### Converts a `GeneralCircuitQuantumState`


```python
from quri_parts.circuit import QuantumCircuit

circuit = QuantumCircuit(2)
circuit.add_H_gate(0)
circuit.add_CNOT_gate(0, 1)

bell_state = quantum_state(n_qubits=2, circuit=circuit)
out_state = evaluate_state_to_vector(bell_state)

print("State vector:")
print(out_state.vector)
print("")
print("Circuit:")
print(out_state.circuit.gates)
```
```output
State vector:
[0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]

Circuit:
()
```
#### Converts a `QuantumStateVector`


```python
import numpy as np
from scipy.stats import unitary_group


circuit = QuantumCircuit(2)
circuit.add_H_gate(0)
circuit.add_CNOT_gate(0, 1)

init_state_vector = unitary_group.rvs(4)[:, 0]
state = quantum_state(n_qubits=2, vector=init_state_vector, circuit=circuit)
out_state = evaluate_state_to_vector(state)

print("State vector:")
print(out_state.vector)
print("")
print("Circuit:")
print(out_state.circuit.gates)
```
```output
State vector:
[ 0.56419181+0.41625524j -0.38655181-0.31055548j -0.12134297+0.14822186j
 -0.34483694-0.32702572j]

Circuit:
()
```
### `run_circuit`

`run_circuit` is a function that acts a quantum circuit on a state vector represented by a `numpy.array` and returns the resulting state vector.


```python
from quri_parts.qulacs.simulator import run_circuit

n_qubits = 2

circuit = QuantumCircuit(n_qubits)
circuit.add_H_gate(0)
circuit.add_CNOT_gate(0, 1)

init_state_vector = unitary_group.rvs(2**n_qubits)[:, 0]
out_state = run_circuit(circuit, init_state_vector)
out_state
```



```output
array([-0.09492785+0.02309928j,  0.09558355-0.02597598j,
        0.3284949 +0.84612921j,  0.36180069-0.16093749j])
```


### `get_marginal_probability`

Sometimes we need to perform partial measurement to a state. `get_marginal_probability` returns the probability of obtaining the specified computational basis eigenstate after measuring a subset of the qubits. For example, for a state with 3 qubits, setting `measured_values={0: 1, 2: 0}` outputs the probability of obtaining 1 from the 0th qubit and 0 from the 2nd qubit. That is, the probability is obtained from the coefficients of $|001\rangle$ and $|011\rangle$.


```python
from quri_parts.qulacs.simulator import get_marginal_probability

n_qubits = 3

init_state = unitary_group.rvs(2**n_qubits)[:, 0]

state = quantum_state(n_qubits=n_qubits, vector=init_state)
out_state = evaluate_state_to_vector(state)
print("State vector:")
print(out_state.vector)
print("")
print(
    "Probability of measuring 1 from to 0th qubit and 0 from the 2nd qubit:",
    get_marginal_probability(init_state, measured_values={0: 1, 2: 0})
)
```
```output
State vector:
[-0.07339477-0.28052222j  0.2207292 +0.07667873j  0.16652878-0.32695901j
  0.26126919+0.20303503j  0.23191247+0.51984261j  0.29447879-0.20130702j
  0.2447073 +0.30430171j -0.11290033+0.02666378j]

Probability of measuring 1 from to 0th qubit and 0 from the 2nd qubit: 0.16408582001654565
```
## Stim simulators

We also provide

- `evaluate_state_to_vector`
- `run_circuit`

in the `quri_parts.stim.simulator` module. The functionalities are the same as those in `quri_parts.qulacs.simulator` except that they only accept Clifford circuits and performs the circuit execution using [Stim](https://github.com/quantumlib/Stim).
