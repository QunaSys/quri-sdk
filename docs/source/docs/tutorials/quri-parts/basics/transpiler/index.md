# Circuit Transpiler

For various reasons, we may want to convert a quantum circuit to another quantum circuit that is semantically equivalent.

For example, if a particular backend supports only a particular gate set, the gate set must be converted. Also, if the qubits are implemented in a particular topology, a conversion may be necessary to make the circuit viable. Converting a semantically equivalent redundant representation to a more concise representation may reduce the execution time of the circuit, the error rate, and the number of qubits.

These motivations can be broadly classified into two categories.

1. Backend (hardware) adaptation
2. Circuit optimization

QURI Parts provides a variety of circuit transpilers for these purposes. Users can also prepare a new transpiler by combining existing transpilers or implementing one from scratch. This tutorial will show you how to handle circuit transpilers with QURI Parts.

## Prerequisite

QURI Parts modules used in this tutorial: `quri-parts-circuit` and `quri-parts-core`. You can install them as follows:


```ipython3
!pip install "quri-parts"
```

## Overview

As an example, let's frist set up the following by circuit and apply the RZ set transpiler. The RZ set transpiler is a transpiler that converts the circuit to one that contains only X, SqrtX, CNOT, and RZ gates. This is done as follows. 


```python
from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.transpile import RZSetTranspiler
from quri_parts.circuit.utils.circuit_drawer import draw_circuit

circuit = QuantumCircuit(3)
circuit.add_H_gate(2)
circuit.add_X_gate(0)
circuit.add_CNOT_gate(2, 1)
circuit.add_Z_gate(2)

print("original:")
draw_circuit(circuit)

transpiler = RZSetTranspiler()
transpiled_circuit = transpiler(circuit)

print("\ntranspiled:")
draw_circuit(transpiled_circuit)
```
```output
original:
   ___                  
  | X |                 
--|1  |-----------------
  |___|                 
           ___          
          |CX |         
----------|2  |---------
          |___|         
   ___      |      ___  
  | H |     |     | Z | 
--|0  |-----●-----|3  |-
  |___|           |___| 

transpiled:
   ___                                  
  | X |                                 
--|3  |---------------------------------
  |___|                                 
                           ___          
                          |CX |         
--------------------------|4  |---------
                          |___|         
   ___     ___     ___      |      ___  
  |RZ |   |sqX|   |RZ |     |     |RZ | 
--|0  |---|1  |---|2  |-----●-----|5  |-
  |___|   |___|   |___|           |___|
```
The `RZSetTranspiler` here is a transpiler made up of multiple simpler transpilers. The goal of this tutorial would be to introduce the transpiler interface and explain how to build customized transpilers.

## Transpiler interface

All transpilers in QURI Parts are `CircuitTranspiler` and can convert `NonParametricQuantumCircuit` to another `NonParametricQuantumCircuit`.


```python
from typing import Callable
from typing_extensions import TypeAlias
from quri_parts.circuit import NonParametricQuantumCircuit

CircuitTranspiler: TypeAlias = Callable[
    [NonParametricQuantumCircuit], NonParametricQuantumCircuit
]
```

There are multiple types of transpiler to perform different kinds of transpilations. They are:

- `GateDecomposer`: A transpiler that decomposes a gate if the gate meets specific condition set by the gate decomposer.
- `GateKindDecomposer`: A transpiler that decomposes a gate for a specific type of gate. In other words, it is a `GateDecomposer` that checks if the gate name matches with the target gate's name.
- `ParallelDecomposer`: A transpiler that composes multiple `GateKindDecomposer`s whose target gates are exclusive of each other. It iterates through the circuit once and decomposes all the type of gates set by the `ParallelDecomposer`.
- `SequentialTranspiler`: A transpiler that composes multiple transpilers and performs the transpilation in sequence.


## Gate kind decomposer and gate decomposer

We first introduce 2 types of basic transpilers that convert gates: `GateKindDecomposer` and `GateDecomposer`.

### `GateDecomposer`

As memtioned above a `GateDecomposer` is a transpiler that decomposes a gate if the gate meets certain conditions. In QURI Parts, two concrete implemetations of them are provided

- `SingleQubitUnitaryMatrix2RYRZTranspiler`
- `TwoQubitUnitaryMatrixKAKTranspiler`

As the names suggest, these gate decomposers decomposes the gate if the gate is a unitary matrix gate acting on 1 qubit or 2 qubits respectively. Let's look at an example with `SingleQubitUnitaryMatrix2RYRZTranspiler`.


```python
from quri_parts.circuit.transpile import SingleQubitUnitaryMatrix2RYRZTranspiler
from scipy.stats import unitary_group

single_qubit_matrix = unitary_group.rvs(2)
double_qubit_matrix = unitary_group.rvs(4)

circuit = QuantumCircuit(2)
circuit.add_UnitaryMatrix_gate([0], single_qubit_matrix)
circuit.add_UnitaryMatrix_gate([0, 1], double_qubit_matrix)


print("original circuit:")
draw_circuit(circuit)

transpiler = SingleQubitUnitaryMatrix2RYRZTranspiler()
transpiled_circuit = transpiler(circuit)

print("")
print("transpiled circuit:")
draw_circuit(transpiled_circuit)
```
```output
original circuit:
   ___     ___  
  |Mat|   |Mat| 
--|0  |---|1  |-
  |___|   |   | 
          |   | 
          |   | 
----------|   |-
          |___| 

transpiled circuit:
   ___     ___     ___     ___  
  |RZ |   |RY |   |RZ |   |Mat| 
--|0  |---|1  |---|2  |---|3  |-
  |___|   |___|   |___|   |   | 
                          |   | 
                          |   | 
--------------------------|   |-
                          |___|
```
From this example above, we see that while both gates are of type `UnitaryMatrix`, but the `GateDecomposer` `SingleQubitUnitaryMatrix2RYRZTranspiler` only takes effect on `UnitaryMatrix` gates acting on a single qubit, thus leaving the 2-qubit unitary matrix gate untouched during the transpilation. A `GateDecomposer` provides a `is_target_gate` to check if a gate is to be converted:


```python
print("Single qubit unitary gate should be converted:", transpiler.is_target_gate(circuit.gates[0]))
print("Double qubit unitary gate should be converted:", transpiler.is_target_gate(circuit.gates[1]))
```
```output
Single qubit unitary gate should be converted: True
Double qubit unitary gate should be converted: False
```
### `GateKindDecomposer`

The other type of basic gate transpiler is the `GateKindDecomposer`. It is a subtype of a `GateDecomposer` that checks if a gate's name matches that of the gate we want to transpile. It does not perform checks on other attributes of a `QuantumGate`. QURI Parts provides an enormous amount of them in the `quri_parts.circuit.transpile.gate_kind_decomposer` module. We suggest you to refer to the [API page](https://quri-parts.qunasys.com/quri_parts/circuit/quri_parts.circuit.transpile.gate_kind_decomposer) for the list of `GateKindDecomposer` we provide.


As an exmaple, we introduce the `H2RZSqrtXTranspiler` that transpiles Hadamard gates to sequence of $R_z$ and $\sqrt{X}$ gates.


```python
from quri_parts.circuit.transpile import H2RZSqrtXTranspiler

circuit = QuantumCircuit(2)
circuit.add_H_gate(0)
circuit.add_X_gate(1)

print("original circuit:")
draw_circuit(circuit)

transpiler = H2RZSqrtXTranspiler()
transpiled_circuit = transpiler(circuit)

print("")
print("transpiled circuit:")
draw_circuit(transpiled_circuit)
```
```output
original circuit:
   ___  
  | H | 
--|0  |-
  |___| 
   ___  
  | X | 
--|1  |-
  |___| 

transpiled circuit:
   ___     ___     ___  
  |RZ |   |sqX|   |RZ | 
--|0  |---|1  |---|2  |-
  |___|   |___|   |___| 
   ___                  
  | X |                 
--|3  |-----------------
  |___|
```
## Sequential transpilers

Multiple transpilers can be applied simply by lining up the transformations. Here, we use a circuit made of a single Toffoli gate as an example. Here we make the following sequence of transpilations
- Transpiler 1:  $\text{Toffoli}$ $\rightarrow$ ($\text{H}$, $\text{T}$, $\text{T}^{\dagger}$, $\text{CNOT}$)
- Transpiler 2:  $\text{H}$ $\rightarrow$ ($R_Z$, $\sqrt{\text{X}}$)
- Transpiler 3:  $\text{T}$ $\rightarrow$ $R_Z$
- Transpiler 4:  $\text{T}^{\dagger}$ $\rightarrow$ $R_Z$

These transpilers are already provided by QURI Parts. Let's demonstrate how to use them:


```python
from quri_parts.circuit.transpile import (
    TOFFOLI2HTTdagCNOTTranspiler,
    H2RZSqrtXTranspiler,
    T2RZTranspiler,
    Tdag2RZTranspiler,
)

circuit = QuantumCircuit(3)
circuit.add_TOFFOLI_gate(0, 1, 2)
print("original:")
draw_circuit(circuit, line_length=120)

circuit = TOFFOLI2HTTdagCNOTTranspiler()(circuit)
circuit = H2RZSqrtXTranspiler()(circuit)
circuit = T2RZTranspiler()(circuit)
circuit = Tdag2RZTranspiler()(circuit)

print("")
print("Sequential transpiled:")
draw_circuit(circuit, line_length=120)
```
```output
original:
        
        
----●---
    |   
    |   
    |   
----●---
    |   
   _|_  
  |TOF| 
--|0  |-
  |___| 

Sequential transpiled:
                                                                                           ___                  
                                                                                          |RZ |                 
--------------------------------------------●-------------------------------●-------●-----|16 |-----●-----------
                                            |                               |       |     |___|     |           
                                            |                      ___      |      _|_     ___     _|_          
                                            |                     |RZ |     |     |CX |   |RZ |   |CX |         
----------------------------●---------------|---------------●-----|10 |-----|-----|15 |---|17 |---|18 |---------
                            |               |               |     |___|     |     |___|   |___|   |___|         
   ___     ___     ___     _|_     ___     _|_     ___     _|_     ___     _|_     ___     ___     ___     ___  
  |RZ |   |sqX|   |RZ |   |CX |   |RZ |   |CX |   |RZ |   |CX |   |RZ |   |CX |   |RZ |   |RZ |   |sqX|   |RZ | 
--|0  |---|1  |---|2  |---|3  |---|4  |---|5  |---|6  |---|7  |---|8  |---|9  |---|11 |---|12 |---|13 |---|14 |-
  |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|
```

It can also be written somewhat more easily by using `SequentialTranspiler` by passing `CircuitTranspiler` instances on creation.


```python
from quri_parts.circuit.transpile import SequentialTranspiler

circuit = QuantumCircuit(3)
circuit.add_TOFFOLI_gate(0, 1, 2)


transpiler = SequentialTranspiler([
    TOFFOLI2HTTdagCNOTTranspiler(),
    H2RZSqrtXTranspiler(),
    T2RZTranspiler(),
    Tdag2RZTranspiler(),
])
circuit = transpiler(circuit)

draw_circuit(circuit, line_length=120)
```
```output
                                                                                           ___                  
                                                                                          |RZ |                 
--------------------------------------------●-------------------------------●-------●-----|16 |-----●-----------
                                            |                               |       |     |___|     |           
                                            |                      ___      |      _|_     ___     _|_          
                                            |                     |RZ |     |     |CX |   |RZ |   |CX |         
----------------------------●---------------|---------------●-----|10 |-----|-----|15 |---|17 |---|18 |---------
                            |               |               |     |___|     |     |___|   |___|   |___|         
   ___     ___     ___     _|_     ___     _|_     ___     _|_     ___     _|_     ___     ___     ___     ___  
  |RZ |   |sqX|   |RZ |   |CX |   |RZ |   |CX |   |RZ |   |CX |   |RZ |   |CX |   |RZ |   |RZ |   |sqX|   |RZ | 
--|0  |---|1  |---|2  |---|3  |---|4  |---|5  |---|6  |---|7  |---|8  |---|9  |---|11 |---|12 |---|13 |---|14 |-
  |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|
```
## Parallel decomposers

It is often the case that we want to transpile multiple types of gates at once. While sequential transpilers can do the job, it is more efficient if we compose multiple `GateKindDecomposer`s into a single `ParallelDecomposer`. We should re-emphasize that a `GateKindDecomposer` is a transpiler that transpile a gate based on what type of gate it is. Hence the gate transformations that makes up a `ParallelDecomposer` should act on gates that are exclusive of each other.


We revisit the last example where we transpile a Toffoli gate into smaller gates. In the last example, we used a sequential transpiler that made up of 4 transpilers. Thus, the circuit was iterated over 4 times. However, if we look at transpilers 2, 3 and 4, the gates that they act on are distinct. Also, any of the output gate sets will not be further transpiled by any other transpilers under consideration. That means it is more desirable to merge the last 3 transpilers into a single `ParallelDecomposer`. This way, the transpilation can be done with 2 iterations to the circuit. To be more explcit, the steps are:

- Step 1: `TOFFOLI2HTTdagCNOTTranspiler`
- Step 2: A `ParallelDecomposer` that consists of:
    - `H2RZSqrtXTranspiler`
    - `T2RZTranspiler`
    - `Tdag2RZTranspiler`

Here we show how we can nest `SequentialTranspiler` and `ParallelDecomposer` to make  a new `CircuitTranspiler`.


```python
from quri_parts.circuit.transpile import ParallelDecomposer, SequentialTranspiler

circuit = QuantumCircuit(3)
circuit.add_TOFFOLI_gate(0, 1, 2)

print("original circuit:")
draw_circuit(circuit)

transpiler = SequentialTranspiler([
    TOFFOLI2HTTdagCNOTTranspiler(),
    ParallelDecomposer([
        H2RZSqrtXTranspiler(),
        T2RZTranspiler(),
        Tdag2RZTranspiler(),
    ]),
])
circuit = transpiler(circuit)

print("\n")
print("transpiled circuit:")
draw_circuit(circuit, line_length=200)
```
```output
original circuit:
        
        
----●---
    |   
    |   
    |   
----●---
    |   
   _|_  
  |TOF| 
--|0  |-
  |___| 


transpiled circuit:
                                                                                           ___                  
                                                                                          |RZ |                 
--------------------------------------------●-------------------------------●-------●-----|16 |-----●-----------
                                            |                               |       |     |___|     |           
                                            |                      ___      |      _|_     ___     _|_          
                                            |                     |RZ |     |     |CX |   |RZ |   |CX |         
----------------------------●---------------|---------------●-----|10 |-----|-----|15 |---|17 |---|18 |---------
                            |               |               |     |___|     |     |___|   |___|   |___|         
   ___     ___     ___     _|_     ___     _|_     ___     _|_     ___     _|_     ___     ___     ___     ___  
  |RZ |   |sqX|   |RZ |   |CX |   |RZ |   |CX |   |RZ |   |CX |   |RZ |   |CX |   |RZ |   |RZ |   |sqX|   |RZ | 
--|0  |---|1  |---|2  |---|3  |---|4  |---|5  |---|6  |---|7  |---|8  |---|9  |---|11 |---|12 |---|13 |---|14 |-
  |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|   |___|
```
## Transpiler for backend adaptation

### Gate set conversion

When a circuit is executed on a real machine in each backend, the gate set of the circuit is often limited to a few universal gates. Also, QURI Parts has high level gate representations such as multi-pauli gates, which are not supported by most backends. Therefore, the circuit must be tranpiled to convert gate set prior to the circuit execution on the backend.

When creating a SamplingBackend or converting a circuit, a default transpiler for each backend is automatically applied, but a user-specified transpiler can be used instead of the default one.

#### Complex gate decomposition

| Module                       | Transpiler                              | Target gate   | Decomposed gate set      |
| ---------------------------- | --------------------------------------- | ------------- | ------------------------ |
| quri_parts.circuit.transpile | PauliDecomposeTranspiler                | Pauli         | \{X, Y, Z\}                |
| quri_parts.circuit.transpile | PauliRotationDecomposeTranspiler        | PauliRotation | \{H, RX, RZ, CNOT\}        |
| quri_parts.circuit.transpile | SingleQubitUnitaryMatrix2RYRZTranspiler | UnitaryMatrix | \{RY, RZ\}                 |
| quri_parts.circuit.transpile | TwoQubitUnitaryMatrixKAKTranspiler      | UnitaryMatrix | \{H, S, RX, RY, RZ, CNOT\} |

#### Gate set conversion

| Module                                  | Transpiler              | Target gate                                                           | Description                                                                     |
| --------------------------------------- | ----------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| quri_parts.circuit.transpile            | RZSetTranspiler         | \{X, SqrtX, RZ, CNOT\}                                                  | Gate set used in superconducting type equipment such as IBM Quantum via Qiskit. |
| quri_parts.circuit.transpile            | RotationSetTranspiler   | \{RX, RY, RZ, CNOT\}                                                    | Intermediate gate set for ion trap type equipment.                              |
| quri_parts.circuit.transpile            | CliffordRZSetTranspiler | \{H, X, Y, Z, S, SqrtX, SqrtXdag, SqrtY, SqrtYdag, Sdag, RZ, CZ, CNOT\} | Clifford + RZ gate set.                                                         |
| quri_parts.quantinuum.circuit.transpile | QuantinuumSetTranspiler | \{U1q, RZ, ZZ, RZZ\}                                                    | Gate set for actual equipment of Quantinuum H1 and H2.                          |
| quri_parts.circuit.transpile            | IonQSetTranspiler       | \{GPi, GPi2, MS\}                                                       | Gate set for actual equipment of IonQ.                                          |



### Qubit mapping

Real devices in the NISQ era are also constrained by the topology of the qubit. In most cases, these constraints are satisfied by the backend automatically transforming the circuit, but sometimes it is desirable to suppress the transformation by the backend and give an explicit mapping of the qubits.

Such qubit mapping can be specified by a dictionary when creating `SamplingBackend`s (see [qubit mapping](../real_devices/sampling_backends/index.md#qubit-mapping) in sampling backends tutorial), but you can also create `QubitRemappingTranspiler` that performs the qubit mapping for given circuits.


```python
from quri_parts.circuit import H, X, CNOT
from quri_parts.circuit.transpile import QubitRemappingTranspiler

circuit = QuantumCircuit(3)
circuit.extend([H(0), X(1), CNOT(1, 2)])

print("original:")
draw_circuit(circuit)

circuit = QubitRemappingTranspiler({0: 2, 1: 0, 2: 1})(circuit)

print("\ntranspiled:")
draw_circuit(circuit)
```
```output
original:
   ___          
  | H |         
--|0  |---------
  |___|         
   ___          
  | X |         
--|1  |-----●---
  |___|     |   
           _|_  
          |CX | 
----------|2  |-
          |___| 

transpiled:
   ___          
  | X |         
--|1  |-----●---
  |___|     |   
           _|_  
          |CX | 
----------|2  |-
          |___| 
   ___          
  | H |         
--|0  |---------
  |___|
```
## Transpiler for circuit optimization

Quantum circuits may be converted to more concise circuits with equivalent action. In actual hardware, certain representations of equivalent circuits may reduce errors or decrease execution time. For example, in the NISQ era, the number of 2-qubit gates often has a significant impact on the error rate, and in the FTQC era, the number of T gates may affect the execution time of a circuit. Optimizing circuits based on these various criteria is another role expected of transpilers.

In QURI Parts, many optimization paths are currently private, but some are available and more will be public in the future.

| Module                              | Transpiler                      | Type                         | Description                                                                                              |
| ----------------------------------- | ------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------- |
| quri_parts.circuit.transpile        | CliffordApproximationTranspiler | Approximate                  | Replace non-Clifford gates with approximate Clifford gate sequences.                                     |
| quri_parts.circuit.transpile        | IdentityInsertionTranspiler     | Equivalent                   | Add Identity gates to qubits which have no gate acting on.                                               |
| quri_parts.circuit.transpile        | IdentityEliminationTranspiler   | Equivalent                   | Remove all Identity gates.                                                                               |
| quri_parts.qiskit.circuit.transpile | QiskitTranspiler                | Equivalent (Numerical error) | Perform backend adaptation, gate set conversion, and circuit simplification using Qiskit’s capabilities. |
| quri_parts.tket.circuit.transpile   | TketTranspiler                  | Equivalent (Numerical error) | Perfomr backend adaptation, gate set conversion, and circuit simplification using Tket’s capabilities.   |

The most basic optimization paths for the rotation gates with parameters are available as follows.

| Module                       | Transpiler                  | Type                         | Description                                                                 |
| ---------------------------- | --------------------------- | ---------------------------- | --------------------------------------------------------------------------- |
| quri_parts.circuit.transpile | FuseRotationTranspiler      | Equivalent (Numerical error) | Fuse consecutive rotation gates of the same kind.                           |
| quri_parts.circuit.transpile | NormalizeRotationTranspiler | Equivalent (Numerical error) | Normalize the rotation angle of the rotation gates to the specified range.  |
| quri_parts.circuit.transpile | RX2NamedTranspiler          | Equivalent (Numerical error) | Convert RX gate if the RX gate is equal to a named gate with no parameters. |
| quri_parts.circuit.transpile | RY2NamedTranspiler          | Equivalent (Numerical error) | Convert RY gate if the RY gate is equal to a named gate with no parameters. |
| quri_parts.circuit.transpile | RZ2NamedTranspiler          | Equivalent (Numerical error) | Convert RZ gate if the RZ gate is equal to a named gate with no parameters. |

## Define your original transpilers

As explained above, a transpiler chained by `SequentialTranspiler` or `ParallellDecomposer` is itself a `CircuitTranspiler` and can be used like other transpilers. In addition, any callable object with an interface of `CircuitTranspiler` can act as a transpiler, whether it is a user defined function or a class.


```python
def transpiler(circuit: NonParametricQuantumCircuit) -> NonParametricQuantumCircuit:
    ...
```

When defining the original transpiler as a class, `CircuitTranspilerProtocol` is defined as an abstract base class that satisfies the properties `CircuitTranspiler` and can be inherited.


```python
from quri_parts.circuit.transpile import CircuitTranspilerProtocol

class Transpiler(CircuitTranspilerProtocol):
    def __call__(self, circuit: NonParametricQuantumCircuit) -> NonParametricQuantumCircuit:
        ...
```

`GateDecomposer` and `GateKindDecomposer` are available for transpilers that convert a specific type of gates in a circuit to some gate sequences (e.g., a transpiler for converting gate sets). `GateDecomposer` can be used to create a new transpiler by writing only the target gate conditions and the transformation of a target gate into a gate sequence. `GateKindDecomposer` is simillar to `GateDecomposer` but it require gate names as target gate conditions.


```python
from collections.abc import Sequence
from quri_parts.circuit import QuantumGate, gate_names
from quri_parts.circuit.transpile import GateDecomposer, GateKindDecomposer

class S0toTTranspiler(GateDecomposer):
    def is_target_gate(self, gate: QuantumGate) -> bool:
        return gate.target_indices[0] == 0 and gate.name == gate_names.S
    
    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gate.T(target), gate.T(target)]
    
class AnyStoTTranspiler(GateKindDecomposer):
    def target_gate_names(self) -> Sequence[str]:
        return [gate_names.S]
    
    def decompose(self, gate: QuantumGate) -> Sequence[QuantumGate]:
        target = gate.target_indices[0]
        return [gate.T(target), gate.T(target)]
```
