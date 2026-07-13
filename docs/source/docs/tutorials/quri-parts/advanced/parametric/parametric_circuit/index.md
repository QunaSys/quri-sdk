# Parametric Circuit

Quantum circuits with variable parameters play an important role in some quantum algorithms, especially variational algorithms. QURI Parts treats such circuits in a special way so that such algorithms can be efficiently performed. In QURI Parts, we provide 2 types of parametric ciricuits. For either type of circuits, the parametric gates are all Pauli rotation gates, but they hold the parameters in different ways. They are

- `UnboundParametricQuantumCircuit`
- `LinearMappedUnboundParametricQuantumCircuit`

In this tutorial, we clearly distinguish two types of parameters: circuit parameter and gate parameter. Circuit parameters refer to independent parameters the circuit holds. Gate parameters refer to the parameter each parametric gate in the circuit hold. Depending on type of the circuit, the gate parameters have different relations with the circuit parameters. In the case of `UnboundParametricQuantumCircuit`, gate parameters are trivial mapped as circuit parameters, i.e. they are the same. In the case of `LinearMappedUnboundParametricQuantumCircuit`, circuit parameters are linearly mapped to gate parameters, so it is generally possible that the number of gate parameters are different from the number of circuit parameters. We will make this clearer with concrete formulae and examples in later sections.

## Prerequisite

QURI Parts modules used in this tutorial: `quri-parts-circuit`, `quri-parts-core` and `quri-parts-qulacs`. You can install them as follows:


```python
# !pip install "quri-parts[qulacs]"
```

## `Parameter` objects

An unbound parameter in a parametric circuit is represented by a `quri_parts.circuit.Parameter` class. A `Parameter` object works as a placeholder for the parameter and does not hold any specific value. Identity of a `Parameter` object is determined by identity of it as a Python object. Even if two `Parameter` objects have the same name, they are treated as different parameters:


```python
from quri_parts.circuit import Parameter, CONST

phi = Parameter("phi")
psi1 = Parameter("psi")
psi2 = Parameter("psi2")

# CONST is a pre-defined parameter that represents a constant.
print(phi, psi1, psi2, CONST)
print("phi == psi1:", phi == psi1)
print("psi1 == psi2:", psi1 == psi2)
print("phi == CONST:", phi == CONST)

print("")
print("Parameters are treated as different even if they have the same name")
print(" Parameter('phi') == Parameter('phi'):", Parameter("phi") == Parameter("phi"))
```
```output
Parameter(name=phi) Parameter(name=psi) Parameter(name=psi2) Parameter(name=)
phi == psi1: False
psi1 == psi2: False
phi == CONST: False

Parameters are treated as different even if they have the same name
 Parameter('phi') == Parameter('phi'): False
```
## Parametric gates

The parametric gates are represented by the `ParametricQuantumGate` object. It is an object containing only the attributes:

- `name`: name of the parametric gate.
- `target_indices`: the qubit it acts on.
- `control_indices`: the qubit that controls the gate on the target qubit.
- `pauli_ids`: The sequence of Pauli matrix labels that represents the label of the Pauli rotation gates.

Like the `QuantumGate` object, it is not to be constructed directly. In QURI Parts, we provide 4 factory functions for creating `ParametricQuantumGates`. They are Pauli rotation gates with no concrete value of rotation angle bound to it.


```python
from quri_parts.circuit import ParametricRX, ParametricRY, ParametricRZ, ParametricPauliRotation

print(ParametricRX(target_index=0))
print(ParametricRY(target_index=0))
print(ParametricRZ(target_index=0))
print(ParametricPauliRotation(target_indices=(0, 1, 2, 3), pauli_ids=(3, 2, 1, 2)))
```
```output
ParametricQuantumGate(name='ParametricRX', target_indices=(0,), control_indices=(), pauli_ids=())
ParametricQuantumGate(name='ParametricRY', target_indices=(0,), control_indices=(), pauli_ids=())
ParametricQuantumGate(name='ParametricRZ', target_indices=(0,), control_indices=(), pauli_ids=())
ParametricQuantumGate(name='ParametricPauliRotation', target_indices=(0, 1, 2, 3), control_indices=(), pauli_ids=(3, 2, 1, 2))
```
## `UnboundParametricQuantumCircuit`

An unbound parametric circuit where each parametric gate in it has its own parameter independent from other parameters. The gates can only depend on a parameter $\theta$ via:

    $$
    \begin{equation}
        \exp\left(- i \frac{\theta}{2} P\right)
    \end{equation}
    $$
where $P$ is any Pauli string. In addition to parametric gates, you can also add all the non-parametric gates supported by the usual `QuantumCircuit` object to a parametric circuit. As an example, let's create a circuit with gates: $[\text{H}_0, \text{CNOT}_{0,1}, \text{RX}(\theta)_0, \text{RY}(\phi)_0, \text{RZ}(\psi)_1]$. Here, $\text{RX}(\theta)$, $\text{RY}(\phi)$ and $\text{RZ}(\psi)$ are rotation gates with parameters independent of each other.


```python
from quri_parts.circuit import UnboundParametricQuantumCircuit
from quri_parts.circuit.utils.circuit_drawer import draw_circuit

parametric_circuit = UnboundParametricQuantumCircuit(2)
parametric_circuit.add_H_gate(0)
parametric_circuit.add_CNOT_gate(0, 1)
p_theta = parametric_circuit.add_ParametricRX_gate(0)
p_phi = parametric_circuit.add_ParametricRY_gate(0)
p_psi = parametric_circuit.add_ParametricRZ_gate(1)

draw_circuit(parametric_circuit)
```
```output
   ___             ___     ___  
  | H |           |PRX|   |PRY| 
--|0  |-----●-----|2  |---|3  |-
  |___|     |     |___|   |___| 
           _|_     ___          
          |CX |   |PRZ|         
----------|1  |---|4  |---------
          |___|   |___|
```
We may check whether or not we have 3 independent parameters added to to circuit and assign values to them. As an example, we assign $\theta = 0.1$, $\phi = 0.2$, $\psi = 0.3$ to the parameters.


```python
print("Number of parameters in the circuit:", parametric_circuit.parameter_count)

# bind parameters:
print("")
print("Bind parameters:")
parametric_circuit.bind_parameters([0.1, 0.2, 0.3]).gates
```
```output
Number of parameters in the circuit: 3

Bind parameters:
```


```output
(QuantumGate(name='H', target_indices=(0,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=()),
 QuantumGate(name='CNOT', target_indices=(1,), control_indices=(0,), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=()),
 QuantumGate(name='RX', target_indices=(0,), control_indices=(), classical_indices=(), params=(0.1,), pauli_ids=(), unitary_matrix=()),
 QuantumGate(name='RY', target_indices=(0,), control_indices=(), classical_indices=(), params=(0.2,), pauli_ids=(), unitary_matrix=()),
 QuantumGate(name='RZ', target_indices=(1,), control_indices=(), classical_indices=(), params=(0.3,), pauli_ids=(), unitary_matrix=()))
```


Looking at the gates, we indeed get the desired circuit with the rotation gates being $\text{RX}(\theta=0.1)_0,\; \text{RY}(\phi=0.2)_0,\; \text{RZ}(\psi=0.3)_1.$


```python
parametric_circuit.gates
```



```output
[QuantumGate(name='H', target_indices=(0,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=()),
 QuantumGate(name='CNOT', target_indices=(1,), control_indices=(0,), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=()),
 ParametricQuantumGate(name='ParametricRX', target_indices=(0,), control_indices=(), pauli_ids=()),
 ParametricQuantumGate(name='ParametricRY', target_indices=(0,), control_indices=(), pauli_ids=()),
 ParametricQuantumGate(name='ParametricRZ', target_indices=(1,), control_indices=(), pauli_ids=())]
```


### Properties

We provide various properties for `UnboundParametricQuantumCircuit`.

#### `qubit_count`:
Number of qubits of the circuit


```python
print("qubit_count:", parametric_circuit.qubit_count)
```
```output
qubit_count: 2
```
#### `depth`:
Depth of the parametric circuit


```python
print("Circuit depth:", parametric_circuit.depth)
```
```output
Circuit depth: 4
```
#### `parameter_count`:
Number of circuit parameters.


```python
print("Parameter count:", parametric_circuit.parameter_count)
```
```output
Parameter count: 3
```
#### `gates`:
All the non-parametric and parametric gates.


```python
print("Gates:")
parametric_circuit.gates
```
```output
Gates:
```


```output
[QuantumGate(name='H', target_indices=(0,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=()),
 QuantumGate(name='CNOT', target_indices=(1,), control_indices=(0,), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=()),
 ParametricQuantumGate(name='ParametricRX', target_indices=(0,), control_indices=(), pauli_ids=()),
 ParametricQuantumGate(name='ParametricRY', target_indices=(0,), control_indices=(), pauli_ids=()),
 ParametricQuantumGate(name='ParametricRZ', target_indices=(1,), control_indices=(), pauli_ids=())]
```


#### `gates_and_params`:
All the non-parametric and parametric gates and the associated circuit parameters. If the gate is non-parametric, the associated parameter would be `None`.


```python
print("Gates and parameters:")
parametric_circuit.gates_and_params
```
```output
Gates and parameters:
```


```output
[(QuantumGate(name='H', target_indices=(0,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=()),
  None),
 (QuantumGate(name='CNOT', target_indices=(1,), control_indices=(0,), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=()),
  None),
 (ParametricQuantumGate(name='ParametricRX', target_indices=(0,), control_indices=(), pauli_ids=()),
  Parameter(name=)),
 (ParametricQuantumGate(name='ParametricRY', target_indices=(0,), control_indices=(), pauli_ids=()),
  Parameter(name=)),
 (ParametricQuantumGate(name='ParametricRZ', target_indices=(1,), control_indices=(), pauli_ids=()),
  Parameter(name=))]
```


### Mutability

An `UnboundParametricQuantumCircuit` is a mutable object where we may freely add gates to. In the case where you want to freeze the parametric circuit like you freeze a `QuantumCircuit`, you may use the `freeze` method to create a new `ImmutableUnboundParametricQuantumCircuit` where in-place gate additions are disabled.


```python
frozen_parametric_circuit = parametric_circuit.freeze()
```

Note that if you freeze the frozen circuit again, you get the same frozen circuit back.


```python
(frozen_parametric_circuit, frozen_parametric_circuit.freeze())
```



```output
(<quri_parts.rust.circuit.circuit_parametric.ImmutableParametricQuantumCircuit at 0x7f7428534870>,
 <quri_parts.rust.circuit.circuit_parametric.ImmutableParametricQuantumCircuit at 0x7f7428534870>)
```


In the case where you want to unfreeze the circuit, you may use the `get_mutable_copy` method.


```python
frozen_parametric_circuit.get_mutable_copy()
```



```output
<quri_parts.rust.circuit.circuit_parametric.ParametricQuantumCircuit at 0x7f74285344b0>
```


#### Mutibality of bound circuit

When we bind the parametric circuit with concrete values, we obtain an `ImmutableBoundParametricQuantumCircuit`. It is an immutable circuit that holds the original parametric circuit and the bound parameters. The parametric circuit held inside it is immutable.


```python
bound_circuit = parametric_circuit.bind_parameters([0.1, 0.2, 0.3])

print("parameter map:")
print(bound_circuit.parameter_map)

print("")
print("unbound circuit:")
print(bound_circuit.unbound_param_circuit)
```
```output
parameter map:
{Parameter(name=): 0.1, Parameter(name=): 0.2, Parameter(name=): 0.3}

unbound circuit:
<quri_parts.rust.circuit.circuit_parametric.ParametricQuantumCircuit object at 0x7f7428534270>
```
### Parameter mapping

Parametric circuits carry `LinearParameterMapping` objects, which denotes how circuit parameters are mapped to gate parameters. In the case of `UnboundParametricQuantumCircuit`, there is no difference between circuit and gate parameters. We may show this with the `has_trivial_parameter_mapping` property.


```python
parametric_circuit.has_trivial_parameter_mapping
```



```output
True
```


Then, let's introduce the `LinearParameterMapping` object stored inside an `UnboundParametricQuantumCircuit`. It can be retrieved by the `.param_mapping` property.


```python
trivial_mapping = parametric_circuit.param_mapping
```

We may retrieve the circuit parameters with the `in_params` property of the `LinearParameterMapping` object. Note that when we use the `.add_Parametric{}_gate()` method while constructing the circuit, we obtain a `Parameter` object. In our example circuit above, they are stored inside the `p_theta`, `p_phi`, `p_psi` variable. We may show that they are the same as the ones retrieved from `.in_params`.


```python
(
    p_theta == parametric_circuit.param_mapping.in_params[0],
    p_phi == parametric_circuit.param_mapping.in_params[1],
    p_psi == parametric_circuit.param_mapping.in_params[2],
)
```



```output
(True, True, True)
```


The gate parameters are represented by `LinearParameterMapping.out_params`. As there is no distinction between the circuit and gate parameters for `UnboundParametricQuantumCircuit`s, we may check that the `p_theta`, `p_phi`, `p_psi` are the same as `.out_params`.


```python
(
    p_theta == parametric_circuit.param_mapping.out_params[0],
    p_phi == parametric_circuit.param_mapping.out_params[1],
    p_psi == parametric_circuit.param_mapping.out_params[2]
)
```



```output
(True, True, True)
```


## `LinearMappedUnboundParametricQuantumCircuit`

An unbound parametric circuit holding a set of independent circuit parameters $\{\theta_i\}$. Different parametric gates generally hold these circuit parameters in the form of different linear functions in the exponent. To be more explicit, a linear mapped parametric gate can look like:

    $$
    \begin{equation}
        \exp\left[- \frac{i}{2} \left(\sum_i a_i \theta_i + b\right) P\right]
    \end{equation}
    $$
where $a_i$ and $b$ are coefficients set by the user. Let's have a quick look at how to construct these circuits. Now, let's create a `LinearMappedUnboundParametricQuantumCircuit` with 2 circuit parameters. $\theta$ and $\phi$. Now, we implement a circuit that depends on the circuit parameters via $[\text{H}_0, \text{CNOT}_{0, 1}, \text{RX}(\theta/2 + \phi/3 + \pi/2)_0, \text{RY}(-\theta/2 + \phi/3)_0, \text{RZ}(\theta/3 - \phi/2 - \pi/2)_1]$. 


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

draw_circuit(linear_param_circuit)
```
```output
   ___             ___     ___  
  | H |           |PRX|   |PRY| 
--|0  |-----●-----|2  |---|3  |-
  |___|     |     |___|   |___| 
           _|_     ___          
          |CX |   |PRZ|         
----------|1  |---|4  |---------
          |___|   |___|
```
In the case of `LinearMappedUnboundParametricQuantumCircuit`, we need to use the `add_parameter` or `add_parameters` methods to assign single or multiple parameters to the circuit. Then, we pass the linear function of the parameters to the `.add_Parametric{}_gate` in the form of a dictionary, whose key is the parameter and value is the corresponding coefficient. Here `quri_parts.circuit.CONST` represents the constant term of a linear function


```python
print("Number of parameters in the circuit:", linear_param_circuit.parameter_count)

# bind parameters:
print("")
print("Bind parameters:")
linear_param_circuit.bind_parameters([0.1, 0.2]).gates
```
```output
Number of parameters in the circuit: 2

Bind parameters:
```


```output
(QuantumGate(name='H', target_indices=(0,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=()),
 QuantumGate(name='CNOT', target_indices=(1,), control_indices=(0,), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=()),
 QuantumGate(name='RX', target_indices=(0,), control_indices=(), classical_indices=(), params=(1.6874629934615633,), pauli_ids=(), unitary_matrix=()),
 QuantumGate(name='RY', target_indices=(0,), control_indices=(), classical_indices=(), params=(0.016666666666666663,), pauli_ids=(), unitary_matrix=()),
 QuantumGate(name='RZ', target_indices=(1,), control_indices=(), classical_indices=(), params=(-1.6374629934615632,), pauli_ids=(), unitary_matrix=()))
```


Finally, it is worth mentioning that we can make the gate parameters become independent of each other, i.e. converting a `LinearMappedUnboundParametricQuantumCircuit` to an `ImmutableUnboundParametricQuantumCircuit`. This is done by the `.primitive_circuit` method.


```python
primitive_circuit = linear_param_circuit.primitive_circuit()
print(primitive_circuit)
print("primitive_circuit is trivially mapped:", primitive_circuit.has_trivial_parameter_mapping)
```
```output
<quri_parts.rust.circuit.circuit_parametric.ImmutableParametricQuantumCircuit object at 0x7f7428536130>
primitive_circuit is trivially mapped: True
```
### Properties

We provide various properties for `LinearMappedUnboundParametricQuantumCircuit`. They are basically the same as those provided by an `UnboundParametricQuantumCircuit`, except that `gates_and_params` is not provided for `LinearMappedUnboundParametricQuantumCircuit`.

#### `qubit_count`:
Number of qubits of the circuit


```python
print("qubit_count:", linear_param_circuit.qubit_count)
```
```output
qubit_count: 2
```
#### `depth`:
Depth of the parametric circuit


```python
print("Circuit depth:", linear_param_circuit.depth)
```
```output
Circuit depth: 4
```
#### `parameter_count`:
Number of circuit parameters.


```python
print("Parameter count:", linear_param_circuit.parameter_count)
```
```output
Parameter count: 2
```
#### `gates`:
All the non-parametric and parametric gates.


```python
print("Gates:")
linear_param_circuit.gates
```
```output
Gates:
```


```output
[QuantumGate(name='H', target_indices=(0,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=()),
 QuantumGate(name='CNOT', target_indices=(1,), control_indices=(0,), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=()),
 ParametricQuantumGate(name='ParametricRX', target_indices=(0,), control_indices=(), pauli_ids=()),
 ParametricQuantumGate(name='ParametricRY', target_indices=(0,), control_indices=(), pauli_ids=()),
 ParametricQuantumGate(name='ParametricRZ', target_indices=(1,), control_indices=(), pauli_ids=())]
```


### Mutability

An `LinearMappedUnboundParametricQuantumCircuit` is a mutable object where we may freely add gates to. In the case where you want to freeze the parametric circuit like you freeze a `QuantumCircuit`, you may use the `freeze` method to create a new `ImmutableLinearMappedUnboundParametricQuantumCircuit` where in-place gate additions are disabled.


```python
frozen_linear_circuit = linear_param_circuit.freeze()
```

Note that if you freeze the frozen circuit again, you get the same frozen circuit back.


```python
(frozen_linear_circuit, frozen_linear_circuit.freeze())
```



```output
(<quri_parts.circuit.circuit_linear_mapped.ImmutableLinearMappedParametricQuantumCircuit at 0x7f742854e4d0>,
 <quri_parts.circuit.circuit_linear_mapped.ImmutableLinearMappedParametricQuantumCircuit at 0x7f742854e4d0>)
```


In the case where you want to unfreeze the circuit, you may use the `get_mutable_copy` method.


```python
frozen_linear_circuit.get_mutable_copy()
```



```output
<quri_parts.circuit.circuit_linear_mapped.LinearMappedParametricQuantumCircuit at 0x7f743e450e90>
```


#### Mutability of bound circuit

When we bind the parametric circuit with concrete values, we obtain an `ImmutableBoundParametricQuantumCircuit`. It is an immutable circuit that holds the original parametric circuit and the bound parameters. The parametric circuit held inside it is immutable. In the case of binding parameters to a linear mapped circuit, the unbound circuit held inside the returned `ImmutableBoundParametricQuantumCircuit` is an `ImmutableUnboundParametricQuantumCircuit` where the parameter mapping is trivial.


```python
bound_linear_circuit = linear_param_circuit.bind_parameters([0.1, 0.2])

print("Parameter map:")
print(bound_linear_circuit.parameter_map)

print("")
print("Unbound circuit:")
print(bound_linear_circuit.unbound_param_circuit)
print("Mapping is trivial:", bound_linear_circuit.unbound_param_circuit.has_trivial_parameter_mapping)
```
```output
Parameter map:
{Parameter(name=): 0.016666666666666663, Parameter(name=): 1.6874629934615633, Parameter(name=): -1.6374629934615632}

Unbound circuit:
<quri_parts.rust.circuit.circuit_parametric.ParametricQuantumCircuit object at 0x7f743e5e3030>
Mapping is trivial: True
```
### Parameter mapping

Now, let's look at the parameter mapping of a `LinearMappedUnboundParametricQuantumCircuit`. As the circuit parameters are mapped to the gate parameters linearly, the `has_trivial_parameter_mapping` property should be `False`.


```python
print(
    "Parameter mapping of LinearMappedUnboundParametricQuantumCircuit is trivial:",
    linear_param_circuit.has_trivial_parameter_mapping
)
```
```output
Parameter mapping of LinearMappedUnboundParametricQuantumCircuit is trivial: False
```
The parameter mapping can be retrieved by the `param_mapping` property. It returns a `LinearParameterMapping` object. Starting from here, we introduce `LinearParameterMapping` in detail.


```python
linear_param_mapping = linear_param_circuit.param_mapping
```

In the `LinearMappedUnboundParametricQuantumCircuit` we created above, there are 2 independent circuit parameters $\theta$ and $\phi$. They can be accessed by the `in_params` property.


```python
linear_param_mapping.in_params
```



```output
(Parameter(name=theta), Parameter(name=phi))
```


As there are 3 parameteric gates that depends on $(\theta, \phi)$, there are 3 gate parameters $\Phi_0$, $\Phi_1$ and $\Phi_2$. They can be accessed by the `out_params` property. They are represented by 3 distinct `Parameter`s with no names assigned to them.


```python
linear_param_mapping.out_params
```



```output
(Parameter(name=), Parameter(name=), Parameter(name=))
```


The circuit parameters are mapped to the gate parameters via:

$$
    \begin{align}
        \Phi_0 &= \frac{\theta}{2} + \frac{\phi}{3} + \frac{\pi}{2}, \nonumber \\
        \Phi_1 &= -\frac{\theta}{2} + \frac{\phi}{3}, \\
        \Phi_2 &= \frac{\theta}{3} - \frac{\phi}{2} - \frac{\pi}{2}, \nonumber\\
    \end{align}
$$

where the linear function can be accessed with the `mapping` property.


```python
linear_param_mapping.mapping
```



```output
{Parameter(name=): mappingproxy({Parameter(name=theta): 0.5,
               Parameter(name=phi): 0.3333333333333333,
               Parameter(name=): 1.5707963267948966}),
 Parameter(name=): mappingproxy({Parameter(name=theta): -0.5,
               Parameter(name=phi): 0.3333333333333333}),
 Parameter(name=): mappingproxy({Parameter(name=theta): 0.3333333333333333,
               Parameter(name=phi): -0.5,
               Parameter(name=): -1.5707963267948966})}
```


We also provide `mapper`s that helps you compute the value of the gate parameters for specified values of circuit parameters. For example, suppose we want to compute gate parameters: 

$$
\begin{equation}
    \{\Phi_0(\theta=0.8, \phi=0.7), \Phi_1(\theta=0.8, \phi=0.7), \Phi_2(\theta=0.8, \phi=0.7)\} \nonumber
\end{equation}
$$

we may write:


```python
linear_param_mapping.mapper({theta: 0.8, phi: 0.7})
```



```output
{Parameter(name=): 2.20412966012823,
 Parameter(name=): -0.1666666666666667,
 Parameter(name=): -1.6541296601282298}
```


Finally, we introduce how to compute the derivatives of the gate parameters with respect to the circuit paramters, i.e.

$$
    \begin{equation}
        \begin{split}
            \left\lbrace\frac{\partial \Phi_0}{\partial \theta}, \frac{\partial \Phi_1}{\partial \theta}, \frac{\partial \Phi_2}{\partial \theta}\right\rbrace \\
            \left\lbrace\frac{\partial \Phi_0}{\partial \phi}, \frac{\partial \Phi_1}{\partial \phi}, \frac{\partial \Phi_2}{\partial \phi}\right\rbrace
        \end{split}
    \end{equation}
$$

This can be done with the `get_derivatives` method.


```python
for p, d in zip(("θ", "φ"), linear_param_mapping.get_derivatives()):
    print(f"Gate parameters' derivatives with respect to {p}: {d.seq_mapper([0.8, 0.7])}")
```
```output
Gate parameters' derivatives with respect to θ: (0.5, -0.5, 0.3333333333333333)
Gate parameters' derivatives with respect to φ: (0.3333333333333333, 0.3333333333333333, -0.5)
```