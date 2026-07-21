# Estimators

In this tutorial, we introduce how to compute expectation values of operators $O$ for a given state $|\psi \rangle$:

$$
\begin{equation}
    \langle O\rangle = \langle\psi|O|\psi\rangle\nonumber
\end{equation}
$$

In QURI Parts, this is done by an `Estimator`. There are various types of `Estimator`s in QURI Parts. In this tutorial we focus on those that computes the exact expectation value of an operator on a pure state. We will also summarize all the currently available estimators in QURI Parts at the bottom of this page.

## Interface

Here we introduce the interface of 2 estimators in QURI Parts: `QuantumEstimator` and `ConcurrentQuantumEstimator`. They are both abstract functions that need concrete implementations for us to perform any computation. Here, we first introduce their definitions and related terminologies:

- `Estimatable`: Represents an `Operator` or a `PauliLabel`.

- `Estimate`:
    An `Estimate` is any object that contains a `value` property and an `error` property.

- `QuantumEstimator`:
    A `QuantumEstimator` is any function that takes an `Estimatable` as its first argument, a `CircuitQuantumState` or `QuantumStateVector` as the second argument and returns an `Estimate`. The `Estimate` represents the estimated expectation value and the error of the estimation.


- `ConcurrentQuantumEstimator`: A `ConcurrentQuantumEstimator` is a function that estimates the expectation values of multiple pairs of `Estimatable` and quantum state.

We demonstrate their function signatures below:



```python
from typing import Union, Callable, Iterable, Sequence
from typing_extensions import TypeAlias, TypeVar
from quri_parts.core.estimator import Estimate
from quri_parts.core.operator import Operator, PauliLabel
from quri_parts.core.state import CircuitQuantumState, QuantumStateVector

#: Represents either CircuitQuantumState or QuantumStateVector.
_StateT = TypeVar("_StateT", bound=Union[CircuitQuantumState, QuantumStateVector])

#: Represents either Operator or PauliLabel.
Estimatable: TypeAlias = Union[Operator, PauliLabel]

#: A function that computes the expectation value of an operator on a given state.
QuantumEstimator: TypeAlias = Callable[[Estimatable, _StateT], Estimate[complex]]

#: A function that computes the expectation values of pairs of operators and states.
ConcurrentQuantumEstimator: TypeAlias = Callable[
    [Sequence[Estimatable], Sequence[_StateT]],
    Iterable[Estimate[complex]],
]
```

## Create and execute estimators

In this section, we introduce concrete instances of `QuantumEstimator`s and `ConcurrentQuantumEstimator`s provided by the `quri_parts.qulacs` package. We will demonstrate how to create them and perform computations with them. The estimators we introduce here are exact estimators, thus the `error` property in the returned `Estimate` will always be 0.

In QURI Parts, we provide several estimator creation functions, they are often named `create_..._estimator`. You would obtain an estimator object by running the creation function. Here, we introduce:

- `create_qulacs_vector_estimator`
- `create_qulacs_vector_concurrent_estimator`

Let's first prepare some operators and states for us to estimate later


```python
import numpy as np
from quri_parts.core.operator import pauli_label, Operator, PAULI_IDENTITY

op1 = Operator({
    pauli_label("X0 Y1"): 2,
    pauli_label("Z0 Y1"): 2j,
    PAULI_IDENTITY: 8,
})
op2 = pauli_label("X0 Y1 Z3")
op3 = pauli_label("X0 X1 X3")
```


```python
from quri_parts.core.state import quantum_state
from quri_parts.circuit import QuantumCircuit, X, CNOT, H

n_qubits = 4
state1 = quantum_state(
    n_qubits, circuit=QuantumCircuit(n_qubits, gates=[X(0), H(1), H(2), CNOT(1, 2)])
)
state2 = quantum_state(n_qubits, bits=0b1101)
state3 = quantum_state(
    n_qubits, vector=np.array(
        [1/np.sqrt(2**n_qubits) for _ in range(2**n_qubits)]
    )
)
```

### Qulacs vector estimator

Here we introduce the vector estimator provided by the `quri_parts.qulacs` package. A vector estimator is an estimator that computes the expectation value of an operator for a pure state exactly, i.e. it computes

$$
\begin{equation}
    \langle O \rangle = \langle \psi|O|\psi \rangle
\end{equation}
$$

with $|\psi\rangle$ being either a `CircuitQuantumState` or a `QuantumStateVector`. Now, let's create a vector estimator:


```python
from quri_parts.qulacs.estimator import create_qulacs_vector_estimator
qulacs_estimator = create_qulacs_vector_estimator()
```

With this vector estimator at hand, we can estimate the expectation values of a an operator for a given state:


```python
print(qulacs_estimator(op1, state1))
print(qulacs_estimator(op2, state2))
print(qulacs_estimator(op3, state3))
```
```output
_Estimate(value=(7.9999999999999964+0j), error=0.0)
_Estimate(value=0j, error=0.0)
_Estimate(value=(1+0j), error=0.0)
```
### Qulacs vector concurrent estimator

There is also a `ConcurrentQuantumEstimator` interface, which estimates multiple operators and multiple states at once. The interface accept either one of the followings:

- One operator, multiple states

- Multiple operators, one state

- The same number of operators and states

First, we create a concurrent estimator


```python
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator
qulacs_concurrent_estimator = create_qulacs_vector_concurrent_estimator()
```

As mentioned above, there are 3 possible inputs to a concurrent estimator

- Estimate concurrently for the same number of operators and states:


```python
qulacs_concurrent_estimator([op1, op2, op3], [state1, state2, state3])
```



```output
[_Estimate(value=(7.9999999999999964+0j), error=0.0),
 _Estimate(value=0j, error=0.0),
 _Estimate(value=(1+0j), error=0.0)]
```


- Estimate concurrently for one operator, multiple states.


```python
qulacs_concurrent_estimator([op1], [state1, state2, state3])
```



```output
[_Estimate(value=(7.9999999999999964+0j), error=0.0),
 _Estimate(value=(8+0j), error=0.0),
 _Estimate(value=(8+0j), error=0.0)]
```


- Estimate concurrently for multiple operators, one state.


```python
qulacs_concurrent_estimator([op1, op2, op3], [state1])
```



```output
[_Estimate(value=(7.9999999999999964+0j), error=0.0),
 _Estimate(value=0j, error=0.0),
 _Estimate(value=0j, error=0.0)]
```


For the case of Qulacs, you can create a `ConcurrentQuantumEstimator` specifying a `concurrent.futures.Executor`(default is `None`, meaning no parallelization) and concurrency (default is 1). Note that since Qulacs itself has multithreading support, using ThreadPoolExecutor or ProcessPoolExecutor may not have performance improvement.


```python
from concurrent.futures import ThreadPoolExecutor
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator

executor = ThreadPoolExecutor(max_workers=4)
qulacs_concurrent_estimator = create_qulacs_vector_concurrent_estimator(executor, concurrency=4)

qulacs_concurrent_estimator([op1, op2, op3], [state1, state2, state3])
```



```output
[_Estimate(value=(7.9999999999999964+0j), error=0.0),
 _Estimate(value=0j, error=0.0),
 _Estimate(value=(1+0j), error=0.0)]
```


## All currently available estimators in QURI Parts

In this section, we summarize all available estimators in QURI Parts. We list out all of them and give brief explanation of what they do. They can be classified into the following groups:

- Quantum estimators
- Parametric estimators
- Overlapping esimtators
- Sampling estimators
- Error mitigation estimators

### Quantum estimators

First, we summarize all the available quantum estimators that performs exact expectation value computations for either noiseless and noisy systems. For the density matrix estimators, we introduce how to use them in the [Noisy Simulation tutorial](../noise_error/index.md).

| Module             | Generating function                                                                                                                                                                                                                                              | Support noisy estimation | Type                                          | Depend on other estimators |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | --------------------------------------------- | -------------------------- |
| quri_parts.qulacs  | [create_qulacs_vector_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/qulacs/quri_parts/qulacs/estimator.py#L73C5-L73C35)([concurrent](https://github.com/QunaSys/quri-parts/blob/main/packages/qulacs/quri_parts/qulacs/estimator.py#L146)) | ✘                        | (Concurrent)QuantumEstimator (State vector)   | ✘                          |
| quri_parts.qulacs  | [create_qulacs_density_matrix_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/qulacs/quri_parts/qulacs/estimator.py#L230)([concurrent](https://github.com/QunaSys/quri-parts/blob/main/packages/qulacs/quri_parts/qulacs/estimator.py#L259)) | ✔                        | (Concurrent)QuantumEstimator (density matrix) | ✘                          |
| quri_parts.itensor | [create_itensor_mps_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/itensor/quri_parts/itensor/estimator.py#L83)([concurrent](https://github.com/QunaSys/quri-parts/blob/main/packages/itensor/quri_parts/itensor/estimator.py#L184))        | ✘                        | (Concurrent)QuantumEstimator (MPS)            | ✘                          |
| quri_parts.stim    | [create_stim_clifford_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/stim/quri_parts/stim/estimator/__init__.py#L59)([concurrent](https://github.com/QunaSys/quri-parts/blob/main/packages/stim/quri_parts/stim/estimator/__init__.py#L130)) | ✘                        | (Concurrent)QuantumEstimator (Clifford)       | ✘                          |

### Parametric estimators

Parametric estimators are estimators that perform expectation value estimation for parametric states. They will be introduced in the [Parametric State tutorial](../../advanced/parametric/estimate_parametric_state/index.md). In short, they compute

$$
\begin{equation}
    \langle O \rangle (\vec{\theta}), \nonumber
\end{equation}
$$

where $\vec{\theta}$ are circuit parameters.

There are 2 additional special parametric estimators in QURI Parts, which are gradient and Hessian estimators. They compute the gradient and Hessian of an expectation value with respect to a set of circuit parameters.

- `GradientEstimator`:

$$
\begin{equation}
    \frac{\partial \langle O \rangle}{\partial \theta_i} (\vec{\theta}) \nonumber
\end{equation}
$$

- `HessianEstimator`:

$$
\begin{equation}
    \frac{\partial^2 \langle O \rangle}{\partial \theta_i \partial \theta_j} (\vec{\theta}) \nonumber
\end{equation}
$$

| Module             | Generating function                                                                                                                                                                                                                                                                                      | Support noisy estimation | Type                                                   | Depend on other estimators    |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------ | ----------------------------- |
| quri_parts.qulacs  | [create_qulacs_vector_parametric_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/qulacs/quri_parts/qulacs/estimator.py#L199)([concurrent](https://github.com/QunaSys/quri-parts/blob/main/packages/qulacs/quri_parts/qulacs/estimator.py#L211))                                      | ✘                        | (Concurrent)ParametricQuantumEstimator(State vector)   | ✘                             |
| quri_parts.qulacs  | [create_qulacs_density_matrix_parametric_estimator](https://github.com/QunaSys/quri-parts-internal/blob/main/packages/qulacs/quri_parts/qulacs/estimator.py#L253) ([concurrent](https://github.com/QunaSys/quri-parts-internal/blob/main/packages/qulacs/quri_parts/qulacs/estimator.py#L304C5-L304C65)) | ✔                        | (Concurrent)ParametricQuantumEstimator(density matrix) | ✘                             |
| quri_parts.itensor | [create_itensor_mps_parametric_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/itensor/quri_parts/itensor/estimator.py#L258) ([concurrent](https://github.com/QunaSys/quri-parts/blob/main/packages/itensor/quri_parts/itensor/estimator.py#L285))                                                                                                 | ✘                        | (Concurrent)ParametricQuantumEstimator (MPS)           | ✘                             |
| quri_parts.core    | [create_numerical_gradient_estimator](https://github.com/QunaSys/quri-parts-internal/blob/main/packages/core/quri_parts/core/estimator/gradient.py#L90)                                                                                                                                                  | ✔ (depends on input estimator)                       | GradientEstimator                                      | ConcurrentParametricEstimator |
| quri_parts.core    | [create_parameter_shift_gradient_estimator](https://github.com/QunaSys/quri-parts-internal/blob/main/packages/core/quri_parts/core/estimator/gradient.py#L189C5-L189C46)                                                                                                                                 | ✔ (depends on input estimator)                        | GradientEstimator                                      | ConcurrentParametricEstimator |
| quri_parts.core    | [create_parameter_shift_hessian_estimator](https://github.com/QunaSys/quri-parts-internal/blob/main/packages/core/quri_parts/core/estimator/hessian.py#L100C5-L100C45)                                                                                                                                   | ✔ (depends on input estimator)                        | HessianEstimator                                       | ConcurrentParametricEstimator |

The "Depend on other estimators" column means that if the estimator is created from another estimator. If yes, we explicitly show what type of estimator should be used to create those estimators that depends on other estimators.

### Overlapping estimators

In addition to those estimators who estimate expectation values, QURI Parts also provides estimators who evaluate the square of the inner product of two states.

- OverlapEstimator: Evaluates: $|\langle\psi|\varphi\rangle|^2$
- OverlapWeightedEstimator: $\sum_{i}w_i|\langle\psi_i|\varphi_i\rangle|^2$
- ParametricOverlapWeightedEstimator: $\sum_{i}w_i|\langle\psi(\vec{\theta}_i)|\varphi(\vec{\phi}_i)\rangle|^2$

| Module                              | Generating function                                                                                                                                                                     | Support noisy estimation | Type                                  | Depend on other estimators |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------- | -------------------------- |
| quri_parts.qulacs.overlap_estimator | [create_qulacs_vector_overlap_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/qulacs/quri_parts/qulacs/overlap_estimator.py#L67C5-L67C45)                           | ✘                        | OverlapEstimator                      | ✘                          |
| quri_parts.qulacs.overlap_estimator | [create_qulacs_vector_overlap_weighted_sum_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/qulacs/quri_parts/qulacs/overlap_estimator.py#L114C5-L114C56)            | ✘                        | OverlapWeightedSumEstimator           | ✘                          |
| quri_parts.qulacs.overlap_estimator | [create_qulacs_vector_parametric_overlap_weighted_sum_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/qulacs/quri_parts/qulacs/overlap_estimator.py#L130C5-L130C67) | ✘                        | ParametricOverlapWeightedSumEstimator | ✘                          |

### Sampling estimators

Sampling estimator is an estimator that estimates the expectation value of an operator with sampling simulations or sampling experiments. Details of sampling estimation will be introduced in the [sampling estimation tutorial](../sampling_estimation/index.md).

| Module          | Generating function                                                                                                                                                                                                                                                                | Support noisy estimation     | Type                         | Depend on other samplers      |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ---------------------------- | ----------------------------- |
| quri_parts.core | [create_sampling_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/core/quri_parts/core/estimator/sampling/estimator.py#L158) ([concurrent](https://github.com/QunaSys/quri-parts/blob/main/packages/core/quri_parts/core/estimator/sampling/estimator.py#L240)) | ✔ (depends on input sampler) | (Concurrent)QuantumEstimator | ConcurrentSampler             |
| quri_parts.core | [create_sampling_overlap_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/core/quri_parts/core/estimator/sampling/overlap_estimator.py#L109)                                                                                                                    | ✔ (depends on input sampler) | OverlapEstimator             | ConcurrentSampler |
| quri_parts.core | [create_sampling_overlap_weighted_sum_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/core/quri_parts/core/estimator/sampling/overlap_estimator.py#L183)                                                                                                       | ✔ (depends on input sampler) | OverlapWeightedSumEstimator  | ConcurrentSampler |

### Error mitigation estimators

We also provide estimators that compute expectation values with a given error mitigation scheme. Their usage are introduced in the [Error Mitigation](../error_mitigation/index.md) tutorial.

| Module                     | Mitigation method        | Generating function                                                                                                                        | Support noisy estimation | Type             | Depend on other estimators                         |
| -------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------ | ---------------- | -------------------------------------------------- |
| quri_parts.algo.mitigation | Clifford Data Regression | [create_cdr_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/mitigation/cdr/cdr.py#L272C5-L272C25) | ✔                        | QuantumEstimator | 1 noiseless and 1 noisy ConcurrentQuantumEstimator |
| quri_parts.algo.mitigation | Zero Noise Extrapolation | [create_zne_estimator](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/mitigation/zne/zne.py#L317)           | ✔                        | QuantumEstimator | Noisy ConcurrentQuantumEstimator |
