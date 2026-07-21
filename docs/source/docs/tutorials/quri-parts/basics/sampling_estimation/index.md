# Sampling Estimation

In order to estimate expectation value of operators on a quantum computer, *sampling measurements* are necessary. In sampling measurements, execution of a quantum circuit and a subsequent measurement of qubits are repeated multiple times. Estimation of expectation value of operators is then performed using statistics of the repeated measurements.

Before performing sampling estimation on real quantum computers, let's see how such procedures can be simulated with a quantum circuit simulator. In this tutorial, we will introduce how to

- Perform grouping for `Operator`s.
- Construct measurement circuit from grouping result.
- Allocate shots for measurement circuits and perform sampling.
- Reconstruct expectation value from sampling result.

## Prerequisite

QURI Parts modules used in this tutorial: `quri-parts-circuit`, `quri-parts-core` and `quri-parts-qulacs`. You can install them as follows:


```ipython3
!pip install "quri-parts[qulacs]"
```

## Overview

The major purpose of this tutorial is to introduce how to perform sampling estimation with a sampling estimator. A sampling estimator is a `QuantumEstimator` where the expectation value is computed via a sampling experiment. As a preparation, we create an operator and a state we want to estimate the expectation value with.


```python
from quri_parts.core.operator import Operator, pauli_label, PAULI_IDENTITY
from quri_parts.core.state import quantum_state

op = Operator({
    pauli_label("Z0"): 0.25,
    pauli_label("Z1 Z2"): 2.0,
    pauli_label("X1 X2"): 0.5 + 0.25j,
    pauli_label("Z1 Y3"): 1.0j,
    pauli_label("Z2 Y3"): 1.5 + 0.5j,
    pauli_label("X1 Y3"): 2.0j,
    PAULI_IDENTITY: 3.0,
})

print("Operator:")
print(op)


state = quantum_state(4, bits=0b0101)
print("")
print("State:")
print(state)
```
```output
Operator:
0.25*Z0 + 2.0*Z1 Z2 + (0.5+0.25j)*X1 X2 + 1j*Z1 Y3 + (1.5+0.5j)*Z2 Y3 + 2j*X1 Y3 + 3.0*I

State:
ComputationalBasisState(qubit_count=4, bits=0b101, phase=0π/2)
```
Here, we provide a sample code for creating and using a sampling estimator. We create a sampling estimator with the following specification:

- Total shots: 5000
- measurement factory: Bitwise commuting Pauli grouping
- Shot allocator: proportional shots allocator

We will explain what "measurement factory" and "Shot allocator" mean in later sections. For now, let's temporarily take them for granted and create a sampling estimator with them:


```python
from quri_parts.core.estimator.sampling import create_sampling_estimator
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.sampling.shots_allocator import create_proportional_shots_allocator
from quri_parts.qulacs.sampler import create_qulacs_vector_concurrent_sampler

total_shots = 5000
concurrent_sampler = create_qulacs_vector_concurrent_sampler()
measurement_factory = bitwise_commuting_pauli_measurement
shots_allocator = create_proportional_shots_allocator()

sampling_estimator = create_sampling_estimator(
    total_shots,
    concurrent_sampler,
    measurement_factory,
    shots_allocator
)
```

Let's look at the result of the sampling estimate:


```python
estimate = sampling_estimator(op, state)
print("Estimated expectation value:", estimate.value)
print("Standard error of this sampling estimation:", estimate.error)
```
```output
Estimated expectation value: (0.7753054753988251-0.032550197113769073j)
Standard error of this sampling estimation: 0.07086365723057407
```
We may compare the sampling estimate's result with the exact result by a vector estimator.


```python
from quri_parts.qulacs.estimator import create_qulacs_vector_estimator

vector_estimator = create_qulacs_vector_estimator()
vector_estimator(op, state)
```



```output
_Estimate(value=(0.75+0j), error=0.0)
```


## Grouping operators into commutable Pauli strings

In sampling experiments, one of the ways to estimate expectation value of such an operator is to estimate the expectation value of each Pauli string and then sum them up. However, it is more efficient to measure multiple Pauli strings at once if they are commutable. The first step is thus to group the Pauli strings into several sets of commutable Pauli strings. This Pauli grouping is an important research subject in context of operator estimation.

Then, from the grouping results, we need to construct a circuit for each commutable group for measurement and sampling, which we call a measurement circuit. The explicit gates involved in a measurement circuit will depend on the content of the commutable group of Pauli strings.

Follow this structure, we use the operator and state constructed above:


```python
print("Operator:")
for pl, coeff in op.items():
    print(f"   + {coeff} * {pl}")

print("")
print("State:", state)
```
```output
Operator:
   + 0.25 * Z0
   + 2.0 * Z1 Z2
   + (0.5+0.25j) * X1 X2
   + 1j * Z1 Y3
   + (1.5+0.5j) * Z2 Y3
   + 2j * X1 Y3
   + 3.0 * I

State: ComputationalBasisState(qubit_count=4, bits=0b101, phase=0π/2)
```
to illustrate how to

- perform operator grouping
- construct measurement circuit
- reconstruct expectation of Pauli strings from measurement results

in QURI Parts. Finally, we will wrap this section up by introducing a convenient object: `CommutablePauliSetMeasurement`, that encapsulates all the steps above.

### Pauli grouping

In QURI Parts, a grouping function is represented by `PauliGrouping`. They are functions that take an `Operator` or a collection of `PauliLabel` into sets of `CommutablePauliSet`, where `CommutablePauliSet` is a set of Pauli strings that commute with each other. The function signature is given by:


```python
from typing import Callable, Set, Union, Iterable
from typing_extensions import TypeAlias
from quri_parts.core.operator import PauliLabel

CommutablePauliSet: TypeAlias = Set[PauliLabel]

#: Represents a grouping function
PauliGrouping = Callable[
    [Union[Operator, Iterable[PauliLabel]]], Iterable[CommutablePauliSet]
]
```

As an example, we use one of the simplest Pauli grouping: *bitwise grouping*, where the groups are determined based on bitwise commutability of the Pauli strings. We can test the grouping as follows:


```python
from quri_parts.core.operator.grouping import bitwise_pauli_grouping
pauli_sets = bitwise_pauli_grouping(op)
print(pauli_sets)
```
```output
frozenset({frozenset({PauliLabel({(1, <SinglePauli.X: 1>), (3, <SinglePauli.Y: 2>)})}), frozenset({PauliLabel()}), frozenset({PauliLabel({(1, <SinglePauli.X: 1>), (2, <SinglePauli.X: 1>)})}), frozenset({PauliLabel({(0, <SinglePauli.Z: 3>)}), PauliLabel({(2, <SinglePauli.Z: 3>), (1, <SinglePauli.Z: 3>)})}), frozenset({PauliLabel({(2, <SinglePauli.Z: 3>), (3, <SinglePauli.Y: 2>)}), PauliLabel({(3, <SinglePauli.Y: 2>), (1, <SinglePauli.Z: 3>)})})})
```
The above looks a bit complicated since the grouping method returns a frozenset of frozensets of Pauli labels.


```python
print(f"Number of groups: {len(pauli_sets)}")
for i, pauli_set in enumerate(pauli_sets):
    labels = ", ".join([str(pauli) for pauli in pauli_set])
    print(f"Group {i} contains: {labels}")
```
```output
Number of groups: 5
Group 0 contains: X1 Y3
Group 1 contains: I
Group 2 contains: X1 X2
Group 3 contains: Z0, Z1 Z2
Group 4 contains: Z2 Y3, Z1 Y3
```
Here, we provide a list of available grouping functions in QURI Parts:

| Grouping function                                                                                                                                   | Explanation                                                           |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [individual_pauli_grouping](https://github.com/QunaSys/quri-parts/blob/main/packages/core/quri_parts/core/operator/grouping/pauli_grouping.py#L27)  | No grouping                                                           |
| [bitwise_pauli_grouping](https://github.com/QunaSys/quri-parts/blob/main/packages/core/quri_parts/core/operator/grouping/pauli_grouping.py#L43)     | Check Pauli string commutability based on bitwise commutability       |
| [sorted_injection_grouping](https://github.com/QunaSys/quri-parts/blob/main/packages/core/quri_parts/core/operator/grouping/pauli_grouping.py#L107) | Explained in [arXiv:1908.06942](https://arxiv.org/abs/1908.06942) |

### Measurement circuit

To perform a measurement for a commutable Pauli set, it is necessary to construct a circuit to be applied before measurement. In the case of bitwise Pauli grouping, the corresponding circuit can be constructed with the `bitwise_commuting_pauli_measurement_circuit` function, which we show as follows:


```python
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement_circuit
from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.utils.circuit_drawer import draw_circuit

pauli_set = {pauli_label("X0 Z2 Y3"), pauli_label("X0 Z1 Y3")}
measurement_circuit = bitwise_commuting_pauli_measurement_circuit(pauli_set)
draw_circuit(QuantumCircuit(4, gates=measurement_circuit))
```
```output
   ___          
  | H |         
--|0  |---------
  |___|         
                
                
----------------
                
                
                
----------------
                
   ___     ___  
  |Sdg|   | H | 
--|1  |---|2  |-
  |___|   |___|
```
### Sampling on a state

In order to measure a commutable Pauli set for a quantum state, the following steps are necessary:

* Construct a circuit that prepares the state
* Concatenate the measurement circuit for the Pauli set after the state preparation circuit
* Perform sampling on the concatenated circuit

Here we use a `ComputationalBasisState` as the initial state for simplicity, though you can use any `CircuitQuantumState`.


```python
from quri_parts.core.state import quantum_state
from quri_parts.qulacs.sampler import create_qulacs_vector_sampler

sampler = create_qulacs_vector_sampler()

initial_state = quantum_state(4, bits=0b0101)

# Circuit for state preparation
state_prep_circuit = initial_state.circuit

# Measurement circuit
pauli_set = {pauli_label("Z2 Y3"), pauli_label("Z1 Y3")}
measurement_circuit = bitwise_commuting_pauli_measurement_circuit(pauli_set)

# Concatenate measurement circuit
sampled_circuit = state_prep_circuit + measurement_circuit
print("State preparation circuit concatenated with measurement circuit:")
draw_circuit(sampled_circuit)

# Sampling
sampling_result = sampler(sampled_circuit, shots=1000)
print("\n")
print("Sampling result:")
print({bin(bits): count for bits, count in sampling_result.items()})
```
```output
State preparation circuit concatenated with measurement circuit:
   ___          
  | X |         
--|0  |---------
  |___|         
                
                
----------------
                
   ___          
  | X |         
--|1  |---------
  |___|         
   ___     ___  
  |Sdg|   | H | 
--|2  |---|3  |-
  |___|   |___| 


Sampling result:
{'0b1101': 516, '0b101': 484}
```
### Reconstructing Pauli string expectation values from sampled values

It is then necessary to read off the measured eigenvalue of a Pauli string from the measurement result. In QURI Parts, this is done with a `PauliReconstructorFactory` and a `PauliReconstructor`. A `PauliReconstructor` is a function that returns the eigenvalue of a Pauli string that corresponds to the measurement result, and a `PauliReconstructorFactory` is a function that creates a `PauliReconstructor` from a `PauliLabel`. Here is their function signatures defined in QURI Parts.



```python
#: PauliReconstructor represents a function that reconstructs a value of a Pauli
#: operator from a measurement result of its measurement circuit.
PauliReconstructor: TypeAlias = Callable[[int], int]

#: PauliReconstructorFactory represents a factory function that returns
#: a :class:`~PauliReconstructor` for a given Pauli operator.
PauliReconstructorFactory: TypeAlias = Callable[[PauliLabel], PauliReconstructor]
```

In the example of the above section, we performed a sampling measurement for $Z_2 Y_3$ and $Z_1 Y_3$, and obtained two bit patterns `0b1101` and `0b0101`. In the case of bitwise Pauli grouping, a value of a Pauli operator can be reconstructed from a sampled bit pattern as follows:


```python
from quri_parts.core.measurement import bitwise_pauli_reconstructor_factory

# Obtain a reconstructor for Z2 Y3
reconstructor = bitwise_pauli_reconstructor_factory(pauli_label("Z2 Y3"))

# Reconstruct a value of Z2 Y3 from a sampled bit pattern 0b1101
pauli_value = reconstructor(0b1101)
print("Measurement result: 0b1101.", f"Corresponding eigenvalue: {pauli_value}")

# Reconstruct from 0b0101
pauli_value = reconstructor(0b0101)
print("Measurement result: 0b0101.", f"Corresponding eigenvalue: {pauli_value}")
```
```output
Measurement result: 0b1101. Corresponding eigenvalue: 1
Measurement result: 0b0101. Corresponding eigenvalue: -1
```
The expectation value of $Z_2 Y_3$ can then be calculated as:


```python
pauli_exp = (
    reconstructor(0b1101) * sampling_result[0b1101] +
    reconstructor(0b0101) * sampling_result[0b0101]
) / 1000
print(pauli_exp)

# Equivalent to above
pauli_exp = sum(
    reconstructor(bits) * count for bits, count in sampling_result.items()
) / sum(sampling_result.values())
print(pauli_exp)
```
```output
0.032
0.032
```
QURI Parts also provide the `trivial_pauli_expectation_estimator` function as a convenient way of obtaining expection value of `PauliLabel` from sampling results.


```python
from quri_parts.core.estimator.sampling import trivial_pauli_expectation_estimator

pauli_exp = trivial_pauli_expectation_estimator(sampling_result, pauli_label("Z2 Y3"))
print(pauli_exp)
```
```output
0.032
```
Here `trivial_pauli_expectation_estimator` can be used because we used bitwise Pauli grouping. In a more general case, `general_pauli_expectation_estimator` should be used with a `PauliReconstructorFactory`:


```python
from quri_parts.core.estimator.sampling import general_pauli_expectation_estimator
pauli_exp = general_pauli_expectation_estimator(
    sampling_result, pauli_label("Z2 Y3"), bitwise_pauli_reconstructor_factory
)
print(pauli_exp)
```
```output
0.032
```
Finally, expectation value of the original operator can be estimated by summing up contributions of each Pauli term. To calculate contribution of $Z_2 Y_3$:


```python
# Coefficient of Z2 Y3 in op
coef = op[pauli_label("Z2 Y3")]
pauli_contrib = coef * pauli_exp
print(pauli_contrib)
```
```output
(0.048+0.016j)
```
Repeating this procedure for all Pauli strings, expectation value of the original operator can be estimated.

### The `CommutablePauliSetMeasurement` object

The above procedure is a bit complicated, so a shortcut method is provided. To use them we first introduce `CommutablePauliSetMeasurement`: it is a data structure containing the following elements:

* `pauli_set`: A set of commutable Pauli operators that are measured together.
* `measurement_circuit`: A circuit required to measure the given `pauli_set`.
* `pauli_reconstructor_factory`: A factory of functions that reconstruct Pauli operator values from sampling result

A set of `CommutablePauliSetMeasurement` needs to be constructed depending on a specific measurement scheme. They are usually obtained by a `CommutablePauliSetMeasurementFactory`, which returns a sequence of `CommutablePauliSetMeasurement` by taking in an `Operator` or a sequence of `PauliLabel`s.


For example, for bitwise Pauli grouping, you can do as follows:


```python
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement

measurements = bitwise_commuting_pauli_measurement(op)
print(f"Number of CommutablePauliSetMeasurement: {len(measurements)}")
measurement = measurements[-1]
```
```output
Number of CommutablePauliSetMeasurement: 5
```
- Commutable Pauli Set


```python
print(", ".join(map(lambda s: str(s), measurement.pauli_set)))
```
```output
Z2 Y3, Z1 Y3
```
- State preparation circuit + Measurement circuit

    We show the circuit for estimating $\langle Z_2 Y_3 \rangle$ and $\langle Z_1 Y_3 \rangle$, and then sample with 1000 shots from it.


```python
full_circuit = state.circuit + measurement.measurement_circuit
draw_circuit(full_circuit)

samling_result = sampler(full_circuit, 1000)
print("\n")
print(f"Measurement Result: {samling_result}")
```
```output
   ___          
  | X |         
--|0  |---------
  |___|         
                
                
----------------
                
   ___          
  | X |         
--|1  |---------
  |___|         
   ___     ___  
  |Sdg|   | H | 
--|2  |---|3  |-
  |___|   |___| 


Measurement Result: Counter({5: 512, 13: 488})
```
- Reconstructor factory

    We retrieve the `PauliReconstructorFactory` from `measurement` and then compute $\langle Z_2 Y_3 \rangle$ and $\langle Z_1 Y_3 \rangle$


```python
print(f"Reconstructor factory: {measurement.pauli_reconstructor_factory}")

for pl in measurement.pauli_set:
    estimate = general_pauli_expectation_estimator(
        sampling_result, pl, measurement.pauli_reconstructor_factory
    )
    print(f" <{pl}> = {estimate}")
    
```
```output
Reconstructor factory: <function bitwise_pauli_reconstructor_factory at 0x7fee00ae3a60>
 <Z2 Y3> = 0.032
 <Z1 Y3> = -0.032
```
We now list out all the `PauliReconstructorFactory`s provided in QURI Parts:


- `bitwise_commuting_pauli_measurement`: Grouping based on bitwise commutation.
- `individual_pauli_measurement`: No grouping.

## Shot Allocators

Up until this point, we have learned how to group an `Operator` into several `CommutablePauliSet`s. We also learned how to evaluate expectation values of _individual_ `PauliLabel` by reading off sampling result from a measurement circuit. The final step necessary for estimation is `PauliSamplingShotsAllocator`: it specifies how total sampling shots should be allocated to measurement of each Pauli sets.


### Interface

In QURI Parts, shot allocators used in sampling estimation are represented by `PauliSamplingShotsAllocator`[^1]. They distribute speficied total shot number to `CommutablePauliSet` according to the `Operator` under consideration or the set of `CommutablePauliSet`. The function signature is given by


```python
from typing import Collection
from quri_parts.core.sampling import PauliSamplingSetting

CommutablePauliSetMeasurement: TypeAlias = Callable[
    [Operator, Collection[CommutablePauliSet], int], Collection[PauliSamplingSetting]
]
```

where the returned `PauliSamplingSetting` is a named tuple holding a `CommutablePauliSet` and the number of shots assigned to it.

[^1]: There is another type of shot allocator `WeightedSamplingShotsAllocator` provided by QURI Parts. However, they are not used for sampling estimation. So, we will not introduce them in this tutorial.

### Available shot allocators

Finally, we introduce the shots allocator available in QURI Parts:


```python
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
    create_proportional_shots_allocator,
    create_weighted_random_shots_allocator,
)
# Allocates shots equally among the Pauli sets
allocator = create_equipartition_shots_allocator()
# Allocates shots proportional to Pauli coefficients in the operator
allocator = create_proportional_shots_allocator()
# Allocates shots using random weights
allocator = create_weighted_random_shots_allocator(seed=777)
```

We also summarize the allocators below:

| Allocator                       | Generating function                                                                                                                                      | Explanation                                                        | Reference                                                                                       |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| equipartition shots allocator   | [create_equipartition_shots_allocator](https://github.com/QunaSys/quri-parts/blob/main/packages/core/quri_parts/core/sampling/shots_allocator.py#L39)    | Allocates shots equally among the Pauli sets                       | -                                                                                               |
| proportional shots allocator    | [create_proportional_shots_allocator](https://github.com/QunaSys/quri-parts/blob/main/packages/core/quri_parts/core/sampling/shots_allocator.py#L73)     | Allocates shots proportional to Pauli coefficients in the operator | - [arXiv:2112.07416 (2021)](https://arxiv.org/pdf/2112.07416.pdf). See Appendix D.<br />- [Phys. Rev. A 92, 042303](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.92.042303)          |
| weighted random shots allocator | [create_weighted_random_shots_allocator](https://github.com/QunaSys/quri-parts/blob/main/packages/core/quri_parts/core/sampling/shots_allocator.py#L122) | Allocates shots using random weights                               | [arXiv:2004.06252 (2020)](https://arxiv.org/abs/2004.06252)                                 |

## Sampling estimation

It has been a long way introducing all the necessary components required to perform sampling estimation. In QURI Parts, we provide a `sampling_estimate` function for you to customize the sampling estimation computation. The required elements for estimating the expectation value of an operator for a `CircuitQuantumState` are:

- Total shots: The total number of shots you use to estimate the expectation value
- Concurrent sampler: A `ConcurrentSampler` that performs the sampling for all (state preparation circuit + measurement circuit).
- Measurement factory: A `CommutablePauliSetMeasurementFactory` that performs grouping for the input operator and evaluates all the expectation values of Pauli strings involved.
- Shots allocator: A `PauliSamplingShotsAllocator` that distributes the total shots among `CommutablePauliSet` generated by the measurement factory.

We now show how to use it.


```python
from quri_parts.qulacs.sampler import create_qulacs_vector_concurrent_sampler
from quri_parts.core.estimator.sampling import sampling_estimate

concurrent_sampler = create_qulacs_vector_concurrent_sampler()

estimate = sampling_estimate(
    op,            # Operator to estimate
    state,         # A circuit state
    5000,          # Total sampling shots
    concurrent_sampler, # ConcurrentSampler
    bitwise_commuting_pauli_measurement, # Factory function for CommutablePauliSetMeasurement
    allocator,     # PauliSamplingShotsAllocator
)
print(f"Estimated expectation value: {estimate.value}")
print(f"Standard error of estimation: {estimate.error}")
```
```output
Estimated expectation value: (0.7778256058066302+0.05988215689349809j)
Standard error of estimation: 0.07112662842561389
```
Most of the time, you might want to embed an estimator or a concurrent estimator into an algorithm. QURI Parts provide `create_sampling_estimator` and `create_sampling_concurrent_estimator` to create `(Concurrent)QuantumEstimator` that can be used like a vector estimator introduced in the [estimator tutorial](../estimators/index.md).


```python
from quri_parts.core.estimator.sampling import create_sampling_estimator
estimator = create_sampling_estimator(
    5000,          # Total sampling shots
    concurrent_sampler, # ConcurrentSampler
    bitwise_commuting_pauli_measurement, # Factory function for CommutablePauliSetMeasurement
    allocator,     # PauliSamplingShotsAllocator
)
estimate = estimator(op, state)
print(f"Estimated expectation value: {estimate.value}")
print(f"Standard error of estimation: {estimate.error}")
```
```output
Estimated expectation value: (0.8045248402391257+0.05988113278846385j)
Standard error of estimation: 0.07079687286344456
```
For concurrent sampling estimators, we create another set of operator and state:


```python
from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.circuit import H, QuantumCircuit

op_1 = op
op_2 = Operator({
    PAULI_IDENTITY: 8,
    pauli_label("X0 Y1 Z2 X3"): 9 + 8j
})

state_1 = state
state_2 = GeneralCircuitQuantumState(4, QuantumCircuit(4, gates=[H(i) for i in range(4)]))
```


```python
from quri_parts.core.estimator.sampling import create_sampling_concurrent_estimator

concurrent_estimator = create_sampling_concurrent_estimator(
    5000,          # Total sampling shots
    concurrent_sampler, # ConcurrentSampler
    bitwise_commuting_pauli_measurement, # Factory function for CommutablePauliSetMeasurement
    allocator,     # PauliSamplingShotsAllocator
)
estimates = concurrent_estimator([op_1, op_2], [state_1, state_2])

print(f"Estimated expectation value 1: {estimates[0].value}")
print(f"Standard error of estimation 1: {estimates[0].error}")
print("")
print(f"Estimated expectation value 2: {estimates[1].value}")
print(f"Standard error of estimation 2: {estimates[1].error}")
```
```output
Estimated expectation value 1: (0.7543098538067212+0.049562397120005305j)
Standard error of estimation 1: 0.07131818403260497

Estimated expectation value 2: (8.1332+0.1184j)
Standard error of estimation 2: 0.1702752120538982
```