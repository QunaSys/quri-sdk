# Variational Algorithms

Recently, *variational quantum algorithms* are actively studied, where optimal values of parameters in parametric quantum circuits are searched. In this section, we see how to construct one of the variational algorithms, *variational quantum eigensolver* (VQE), using the gradient.

*Variational quantum eigensolver* (VQE) is a method to optimize an expectation value of an operator (e.g. energy of a molecule) over parametrized quantum states. There are two major components in VQE:

- *Ansatz*: A parametric quantum circuit which generates the parametrized quantum states subject to optimization
- *Optimizer*: A method to numerically optimize the expectation value of the operator

## Ansatz

In context of VQE, ansatz refers to a parametric quantum circuit used for generating parametrized quantum states for which expectation values of the target operator is evaluated. You can define a `(LinearMapped)UnboundParametricQuantumCircuit` on your own, or use a well-known ansatz defined in `quri_parts.algo.ansatz` package. In this example we use a hardware-efficient ansatz[^Kandala2017]:

[^Kandala2017]: Kandala, A., Mezzacapo, A., Temme, K. et al. Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. [Nature **549**, 242–246 (2017)](https://doi.org/10.1038/nature23879).


```python
from quri_parts.algo.ansatz import HardwareEfficient

hw_ansatz = HardwareEfficient(qubit_count=4, reps=3)
```

In order to evaluate the expectation value, the parametrized quantum state is necessary, which is obtained by applying the ansatz to a specific initial state. Here we use a computational basis state $|0011\rangle$.


```python
from quri_parts.core.state import quantum_state, apply_circuit

cb_state = quantum_state(4, bits=0b0011)
parametric_state = apply_circuit(hw_ansatz, cb_state)
```

### List of available ansatz

Here we list out all the available ansatz provided by `quri_parts.algo.ansatz`. There are more chemistry-related ansatz provided in `quri_parts.chem.ansatz` and `quri_parts.openfermion.ansatz`, which are introduced in the quantum chemistry tutorial.

| Ansatz                                                                                                                                     | Reference                                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [HardwareEfficient](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/ansatz/hardware_efficient.py#L46)        | [Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets](https://www.nature.com/articles/nature23879)                      |
| [HardwareEfficientReal](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/ansatz/hardware_efficient.py#L128)   | [Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets](https://www.nature.com/articles/nature23879)                      |
| [SymmetryPreserving](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/ansatz/symmetry_preserving.py#L62)      | [Efficient Symmetry-Peserving State Preparation Circuits for the Variational Quantum Eigensolver Algorithm](https://www.nature.com/articles/s41534-019-0240-1) |
| [SymmetryPreservingReal](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/ansatz/symmetry_preserving.py#L113) | [Calculating transition amplitudes by variational quantum deflation](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.013173)            |
| [TwoLocal](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/ansatz/two_local.py#L63)                          |                                                                                                                                                                |

## Optimizer

An optimizer searches optimal parameters that minimize a given cost function. In context of VQE, the cost function is the expectation value of the target operator. Some optimizers use only the cost function itself, while others use gradient of the cost function for efficient optimization. You can use optimizers provided by libraries such as `scipy.optimize`, or ones provided in `quri_parts.algo.optimizer` package. In this example we use Adam[^Kingma2014], which uses the gradient.

[^Kingma2014]: Diederik P. Kingma, Jimmy Ba, Adam: A Method for Stochastic Optimization. [arXiv:1412.6980 (2014)](https://doi.org/10.48550/arXiv.1412.6980)


```python
from quri_parts.algo.optimizer import Adam

# You can pass optional parameters. See the reference for details
adam_optimizer = Adam()
```

### List of available optimizers

Here, we list out all the optimizers available in QURI Parts.

| Optimizer Name                                                                                                    | Reference                                                                                                                                                |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [AdaBelief](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/optimizer/adam.py#L170) | [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/abs/2010.07468)                                          |
| [Adam](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/optimizer/adam.py#L48)       | [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)                                                                           |
| [NFT](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/optimizer/nft.py#L118)        | [Sequential minimal optimization for quantum-classical hybrid algorithms](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043158) |
| [NFTfit](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/optimizer/nft.py#L201)     | [Sequential minimal optimization for quantum-classical hybrid algorithms](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043158) |
| [LBFGS](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/optimizer/lbfgs.py#L59)     | [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5)                                                                       |
| [SPSA](https://github.com/QunaSys/quri-parts/blob/main/packages/algo/quri_parts/algo/optimizer/spsa.py#L34)       | [Implementation of the simultaneous perturbation algorithm for stochastic optimization](https://ieeexplore.ieee.org/document/705889)                     |

## Running VQE

We first define a target operator, whose expectation value is subject to the optimization:


```python
from quri_parts.core.operator import Operator, pauli_label, PAULI_IDENTITY

# This is Jordan-Wigner transformed Hamiltonian of a hydrogen molecule
hamiltonian = Operator({
    PAULI_IDENTITY: 0.03775110394645542,
    pauli_label("Z0"): 0.18601648886230593,
    pauli_label("Z1"): 0.18601648886230593,
    pauli_label("Z2"): -0.2694169314163197,
    pauli_label("Z3"): -0.2694169314163197,
    pauli_label("Z0 Z1"): 0.172976101307451,
    pauli_label("Z0 Z2"): 0.12584136558006326,
    pauli_label("Z0 Z3"): 0.16992097848261506,
    pauli_label("Z1 Z2"): 0.16992097848261506,
    pauli_label("Z1 Z3"): 0.12584136558006326,
    pauli_label("Z2 Z3"): 0.17866777775953396,
    pauli_label("X0 X1 Y2 Y3"): -0.044079612902551774,
    pauli_label("X0 Y1 Y2 X3"): 0.044079612902551774,
    pauli_label("Y0 X1 X2 Y3"): 0.044079612902551774,
    pauli_label("Y0 Y1 X2 X3"): -0.044079612902551774,
})
```

Using this operator and the parametric state prepared above, we can define the cost function as a function of the circuit parameters:


```python
from typing import Sequence
from quri_parts.qulacs.estimator import create_qulacs_vector_parametric_estimator

estimator = create_qulacs_vector_parametric_estimator()

def cost_fn(param_values: Sequence[float]) -> float:
    estimate = estimator(hamiltonian, parametric_state, param_values)
    return estimate.value.real
```

We also define gradient of the cost function using numerical gradient:


```python
import numpy as np
from quri_parts.core.estimator.gradient import create_numerical_gradient_estimator
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_parametric_estimator

qulacs_concurrent_parametric_estimator = create_qulacs_vector_concurrent_parametric_estimator()
gradient_estimator = create_numerical_gradient_estimator(
    qulacs_concurrent_parametric_estimator,
    delta=1e-4,
)

def grad_fn(param_values: Sequence[float]) -> Sequence[float]:
    estimate = gradient_estimator(hamiltonian, parametric_state, param_values)
    return np.asarray([g.real for g in estimate.values])
```

Then we can run VQE with a QURI Parts optimizer:


```python
from quri_parts.algo.optimizer import (
    OptimizerStatus, Optimizer, OptimizerState, CostFunction, GradientFunction
)

def vqe(
    init_params: Sequence[float],
    cost_fn: CostFunction,
    grad_fn: GradientFunction,
    optimizer: Optimizer
) -> OptimizerState:
    opt_state = optimizer.get_init_state(init_params)
    while True:
        opt_state = optimizer.step(opt_state, cost_fn, grad_fn)
        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
    return opt_state

init_params = [0.1] * hw_ansatz.parameter_count
result = vqe(init_params, cost_fn, grad_fn, adam_optimizer)
print("Optimized value:", result.cost)
print("Optimized parameter:", result.params)
print("Iterations:", result.niter)
print("Cost function calls:", result.funcalls)
print("Gradient function calls:", result.gradcalls)
```
```output
Optimizer converged
Optimized value: -1.1119813406104204
Optimized parameter: [ 5.47178290e-02  8.40762190e-02  5.12253347e-02  8.19750366e-02
 -9.72099704e-03 -1.16141832e-01 -3.06727507e-03  9.66792840e-01
  1.27323903e-01  1.04790837e-01  1.27097746e-01  9.40512719e-02
 -1.60419268e-02  9.92326531e-01 -3.35897820e-02  9.91027219e-01
  6.44048149e-02  2.49942838e-04  6.43611654e-02 -5.72089598e-03
 -1.48640069e-02 -1.16555427e-01 -3.59503991e-02  9.79005522e-01
  1.67652639e-02 -2.35033759e-01  1.34115103e-02 -2.24492856e-01
 -2.91851967e-02  4.35033714e-01 -3.52284761e-03  4.24493014e-01]
Iterations: 24
Cost function calls: 25
Gradient function calls: 24
```
You can also run VQE with a SciPy optimizer:


```python
from scipy.optimize import minimize, OptimizeResult
from typing import Any

def vqe_scipy(
    init_params: Sequence[float],
    cost_fn: CostFunction,
    grad_fn: GradientFunction, 
    method: str,
    options: dict[str, Any]
) -> OptimizeResult:
    return minimize(cost_fn, init_params, jac=grad_fn, method=method, options=options)

init_params = [0.1] * hw_ansatz.parameter_count
bfgs_options = {
    "gtol": 1e-6,
}
result = vqe_scipy(init_params, cost_fn, grad_fn, "BFGS", bfgs_options)
print(result.message)
print("Optimized value:", result.fun)
print("Optimized parameter:", result.x)
print("Iterations:", result.nit)
print("Cost function calls:", result.nfev)
print("Gradient function calls:", result.njev)
```
```output
Optimization terminated successfully.
Optimized value: -1.1299047840976442
Optimized parameter: [ 2.05593646e-03  3.87341918e-02  6.61727563e-01  2.42278307e-03
  3.12738525e-01 -4.21253119e-02 -1.39600146e+00 -2.85039706e-03
  3.35892634e-01  2.15408237e-04  6.57480272e-01 -2.74163479e-01
  6.78252570e-01  1.19267986e-01  2.20687940e+00 -5.74165023e-03
  1.56985600e+00  6.69304284e-06 -1.14405759e-03  1.69116946e-01
  1.79284277e-01 -1.46789991e-01 -2.27287648e-01  1.55406107e-02
  1.23560233e+00  1.09112774e-01 -1.55674962e-03  1.09133237e-01
 -5.23284705e-01  9.08872954e-02 -6.40333420e-01  9.08667616e-02]
Iterations: 175
Cost function calls: 182
Gradient function calls: 182
```