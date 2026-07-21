# Gradient Estimators

One of the most important element of performing optimization algorithms is computing the gradient of a physical observable's gradient with respect to a set of circuit parameters:

$$
    \begin{equation}
        \frac{\partial \langle O\rangle (\vec{\theta})}{\partial \theta_i}
    \end{equation}
$$
In this tutorial, we introduce the gradient estimators provided by QURI Parts. They are:

- Numerical gradient estimator: A gradient estimator that estimates the gradient based on finite difference method.
- Parameter shift gradient estimator: A gradient estimator that estimates the gradient based on the parameter shift method.

## Prerequisite
QURI Parts modules used in this tutorial: `quri-parts-circuit`, `quri-parts-core`, and `quri-parts-qulacs`. You can install them as follows:



```ipython3
!pip install "quri-parts[qulacs]"
```

## Interface

A gradient estimator is represented by the `GradientEstimator` interface. It represents a function that estimates gradient values of an expectation value of a given `Operator` for a given parametric state with given parameter values (the third argument). It's function signature is


```python
from typing import Callable, Sequence, Union
from typing_extensions import TypeAlias, TypeVar
from quri_parts.core.estimator import Estimatable, Estimates
from quri_parts.core.state import ParametricCircuitQuantumState, ParametricQuantumStateVector

# Generic type of parametric states
_ParametricStateT = TypeVar(
    "_ParametricStateT",
    bound=Union[ParametricCircuitQuantumState, ParametricQuantumStateVector],
)

# Function signature of a `GradientEstimator` defined in QURI Parts.
GradientEstimator: TypeAlias = Callable[
    [Estimatable, _ParametricStateT, Sequence[float]],
    Estimates[complex],
]
```

You may create a `GradientEstimator` from a generating function. They are often named as `create_..._gradient_estimator`. To create a `GradientEstimator`, you need to pass in a `ConcurrentParametricQuantumEstimator` to the generating function. Here, we use the one provided by `quri_parts.qulacs`


```python
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_parametric_estimator
concurrent_parametric_estimator = create_qulacs_vector_concurrent_parametric_estimator()
```

## Preparation

Let's prepare the operator and the parametric state we use through out this tutorial.


```python
from quri_parts.core.operator import Operator, pauli_label

operator = Operator({
    pauli_label("X0 Y1"): 0.5,
    pauli_label("Z0 X1"): 0.2,
})
```

The linear mapping of the parametric circuit is slightly different from previous sections. Here, the circuit parameter and gate parameters are related via:

$$
\begin{equation}
    \begin{split}
        \Theta_1 &= \left(\frac{\theta}{2} + \frac{\phi}{2} + \frac{1}{2}\right)\pi \\
        \Theta_2 &= \left(-\frac{\theta}{2} + \frac{\phi}{3}\right)\pi \\
        \Theta_3 &= \left(\frac{\theta}{3} - \frac{\phi}{2} -  \frac{1}{2}\right)\pi
    \end{split}
\end{equation}
$$

for aesthetical reason when we discuss the details of the parameter shift rule later.


```python
import numpy as np
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit, CONST
from quri_parts.core.state import quantum_state

n_qubits = 2
linear_param_circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubits)
theta, phi = linear_param_circuit.add_parameters("theta", "phi")

linear_param_circuit.add_H_gate(0)
linear_param_circuit.add_CNOT_gate(0, 1)
linear_param_circuit.add_ParametricRX_gate(0, {theta: np.pi/2, phi: np.pi/3, CONST: np.pi/2})
linear_param_circuit.add_ParametricRY_gate(0, {theta: -np.pi/2, phi: np.pi/3})
linear_param_circuit.add_ParametricRZ_gate(1, {theta: np.pi/3, phi: -np.pi/2, CONST: -np.pi/2})

param_state = quantum_state(n_qubits, circuit=linear_param_circuit)
```

## Numerical gradient estimator

The numerical gradient estimator computes the gradient according to the [finite difference method](https://en.wikipedia.org/wiki/Finite_difference_method), i.e.

$$
\begin{equation}
    \frac{\partial f}{\partial \theta_i} = \frac{f(\theta_i + \delta) - f(\theta_i - \delta)}{2\delta}
\end{equation}
$$
with $\delta$ being a small number we can freely set. Thus, to create a numerical gradient estimator, we need to pass in $\delta$ along with the concurrent parametric estimator.


```python
from quri_parts.core.estimator.gradient import create_numerical_gradient_estimator

numerical_gradient_estimator = create_numerical_gradient_estimator(
    concurrent_parametric_estimator,
    delta=1e-10
)
```

Now, we may estimate the gradient of the parametric state on $\theta = 0.1,\; \phi = 0.2$.


```python
numerical_gradient_estimator(operator, param_state, [0.1, 0.2]).values
```



```output
[(-0.3508326962275987+0j), (0.5306499684110122+0j)]
```


## Parameter shift gradient estimator

The parameter shift rule was introduced in the cited paper below [^Mitarai2018]. As a very quick review, we may write the parameter shift rule as:


$$
\begin{equation}
    \frac{\partial f}{ \partial \theta_i} = \sum_{a} \frac{\partial f}{\partial \Theta_a} \frac{\partial \Theta_a}{\partial \theta_i} = \frac{1}{2}\sum_{a} \left[ f(\Theta_a + \frac{\pi}{2}) - f(\Theta_a - \frac{\pi}{2}) \right]\frac{\partial \Theta_a}{\partial \theta_i}
\end{equation}
$$

where $f$ is the expectation value of any operator estimated on a circuit state, $\Theta_a$ are independent gate parameters and $\theta_i$ are the independent circuit parameters. 


To create a parameter shift gradient estimator, we only need to pass in the concurremt parametric estimator to the generating function: `create_parameter_shift_gradient_estimator`.


```python
from quri_parts.core.estimator.gradient import create_parameter_shift_gradient_estimator

param_shift_gradient_estimator = create_parameter_shift_gradient_estimator(
    concurrent_parametric_estimator,
)
```

Now, we may estimate the gradient of the parametric state on $\theta = 0.1,\; \phi = 0.2$.


```python
param_shift_gradient_estimator(operator, param_state, [0.1, 0.2]).values
```



```output
[(-0.3508320725634084+0j), (0.5306488303307602+0j)]
```


We can see that the result is very close to the one estimated by the numerical gradient estimator.

### Explanation of how gradient evaluation by parameter shift rule works

When evaluating the gradient with parameter shift rule, parameters of each parametric gates need to be shifted independently, even if they depend on the same circuit parameters. It is also necessary to compute derivative of each gate parameter with respect to the circuit parameters so that we can use chain rule of differentiation. Therefore we need the followings:

- The parametric circuit where each gate parameters are treated as independent (`UnboundParametricQuantumCircuit` in QURI Parts).
- Parameter shifts for each gate parameters for each circuit parameters.
- Differential coefficients corresponding to each parameter shifts.


```python
from quri_parts.circuit.parameter_shift import ShiftedParameters
from quri_parts.core.state import ParametricCircuitQuantumState
from typing import Sequence, Collection

def get_raw_param_state_and_shifted_parameters(
    state: ParametricCircuitQuantumState,
    params: Sequence[float]
) -> tuple[ParametricCircuitQuantumState, Collection[tuple[Sequence[float], float]]]:
    
    param_mapping = state.parametric_circuit.param_mapping
    raw_circuit = state.parametric_circuit.primitive_circuit()
    parameter_shift = ShiftedParameters(param_mapping)
    derivatives = parameter_shift.get_derivatives()
    shifted_parameters = [
        d.get_shifted_parameters_and_coef(params) for d in derivatives
    ]

    raw_param_state = ParametricCircuitQuantumState(state.qubit_count, raw_circuit)

    return raw_param_state, shifted_parameters
```

Here, the returned `raw_param_state` is the parametric circuit quantum state holding a parametric circuit with all of its parameters independent of each other. `shifted_parameters` holds:

$$
\begin{equation}
\left\lbrace
\left(\Theta_0, \cdots ,\Theta_a \pm \frac{\pi}{2}, \cdots ,\Theta_{N_\text{gates}-1}\right), \pm\frac{1}{2}\frac{\partial \Theta_a}{\partial \theta_i}
\right\rbrace
\end{equation}
$$

For example, let's look at the shifted parameters and coefficients with circuit parameters $\theta = 0.1,\; \phi = 0.2$. In the linear mapped circuit we constructed above, the circuit parameter and gate parameters are related via:

$$
\begin{equation}
    \begin{split}
        \Theta_1 &= \left(\frac{\theta}{2} + \frac{\phi}{2} + \frac{1}{2}\right)\pi \\
        \Theta_2 &= \left(-\frac{\theta}{2} + \frac{\phi}{3}\right)\pi \\
        \Theta_3 &= \left(\frac{\theta}{3} - \frac{\phi}{2} -  \frac{1}{2}\right)\pi
    \end{split}
\end{equation}
$$


```python
raw_state, shifted_params_and_coefs = get_raw_param_state_and_shifted_parameters(
    param_state, [0.1, 0.2]
)
bound_circuit = param_state.parametric_circuit.bind_parameters([0.1, 0.2]).parameter_map
gate_parameters = np.array(list(bound_circuit.values()))
gate_param_str = ", ".join(map(lambda f: str(np.round(f/np.pi, 3)) + "π", gate_parameters)) 
print(f"Gate parameters: ({gate_param_str})")

for i, params_and_coefs in enumerate(shifted_params_and_coefs):
    print("")
    print(f"Parameter shifts for circuit parameter {i}:")
    for p, c in params_and_coefs:
        p_str = ", ".join(map(lambda f: str(np.round(f/np.pi, 3)) + "π", p))
        diff = np.array(p) - gate_parameters
        p_str = ", ".join(map(lambda f: str(np.round(f/np.pi, 3)) + "π", diff))
        print(f"  gate params:  ({gate_param_str}) + ({p_str}), coefficient: {c/np.pi: .3f}π")
```
```output
Gate parameters: (-0.567π, 0.017π, 0.617π)

Parameter shifts for circuit parameter 0:
  gate params:  (-0.567π, 0.017π, 0.617π) + (1.683π, 0.0π, -1.183π), coefficient:  0.250π
  gate params:  (-0.567π, 0.017π, 0.617π) + (1.183π, 0.0π, -1.683π), coefficient: -0.167π
  gate params:  (-0.567π, 0.017π, 0.617π) + (1.183π, 0.5π, -1.183π), coefficient: -0.250π
  gate params:  (-0.567π, 0.017π, 0.617π) + (1.183π, 0.0π, -0.683π), coefficient:  0.167π
  gate params:  (-0.567π, 0.017π, 0.617π) + (1.183π, -0.5π, -1.183π), coefficient:  0.250π
  gate params:  (-0.567π, 0.017π, 0.617π) + (0.683π, 0.0π, -1.183π), coefficient: -0.250π

Parameter shifts for circuit parameter 1:
  gate params:  (-0.567π, 0.017π, 0.617π) + (1.183π, 0.0π, -0.683π), coefficient: -0.250π
  gate params:  (-0.567π, 0.017π, 0.617π) + (0.683π, 0.0π, -1.183π), coefficient: -0.167π
  gate params:  (-0.567π, 0.017π, 0.617π) + (1.183π, -0.5π, -1.183π), coefficient: -0.167π
  gate params:  (-0.567π, 0.017π, 0.617π) + (1.683π, 0.0π, -1.183π), coefficient:  0.167π
  gate params:  (-0.567π, 0.017π, 0.617π) + (1.183π, 0.0π, -1.683π), coefficient:  0.250π
  gate params:  (-0.567π, 0.017π, 0.617π) + (1.183π, 0.5π, -1.183π), coefficient:  0.167π
```
We then obtain the gradient by 
1. estimating the expectation value of the operator for each shifted gate parameters
2. sum them up with the corresponding coefficients multiplied.

This can be done as follows:


```python
from quri_parts.core.estimator import Estimatable

def get_parameter_shift_gradient(
    op: Estimatable,
    raw_state: ParametricCircuitQuantumState,
    shifted_params_and_coefs
) -> list[complex]:
    # Collect gate parameters to be evaluated
    gate_params = set()
    for params_and_coefs in shifted_params_and_coefs:
        for p, _ in params_and_coefs:
            gate_params.add(p)
    gate_params_list = list(gate_params)

    # Prepare a parametric estimator
    estimator = create_qulacs_vector_concurrent_parametric_estimator()
    
    # Estimate the expectation values
    estimates = estimator(op, raw_state, gate_params_list)
    estimates_dict = dict(zip(gate_params_list, estimates))
    
    # Sum up the expectation values with the coefficients multiplied
    gradient = []
    for params_and_coefs in shifted_params_and_coefs:
        g = 0.0
        for p, c in params_and_coefs:
            g += estimates_dict[p].value * c
        gradient.append(g)
    
    return gradient

# Example
gradient = get_parameter_shift_gradient(operator, raw_state, shifted_params_and_coefs)
print("Estimated gradient:", gradient)
```
```output
Estimated gradient: [(-0.3508320725634084+0j), (0.5306488303307602+0j)]
```
[^Mitarai2018]: Mitarai, K. and Negoro, M. and Kitagawa, M. and Fujii, K., [Phys. Rev. A **98**, 032309 (2018)](https://doi.org/10.1103/PhysRevA.98.032309). [arXiv:1803.00745](https://arxiv.org/abs/1803.00745).
