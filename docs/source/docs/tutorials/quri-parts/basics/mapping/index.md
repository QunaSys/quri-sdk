# Fermion-qubit Mappings

In order to simulate the dynamics of physical systems with a quantum computer, it is necessary to map the Hamiltonian of an electron to the qubit counterpart. Hamiltonians for fermionic systems, as typically used in quantum chemistry, are often expressed using anti-commuting creation and annihilation operators: $c_i^{\dagger}$, $c_i$ under second quantization. If we can rewrite the creation and annihilation operators as Pauli operators that can act on qubits, we can represent them on a quantum computer.

Here, $c_i$, $c_i^{\dagger}$ satisfy the anti-commutation relations:

$$
\begin{align}
    \{c_{i}, c_{j}^{\dagger}\} &= \delta_{ij}\\
    \{c_{i}, c_{j}\} &= 0\\
    \{c_{i}^{\dagger}, c_{j}^{\dagger}\} &= 0
\end{align}
$$

and $i$, $j$ denote the label of degree of freedom the operator acts on. 

Fermionic wavefunctions exhibit antisymmetry, but when mapping directly from spin orbitals to qubits on a quantum computer, where the presence of an electron in a spin orbital is represented as $|1\rangle$ and the absence as $|0\rangle$, this antisymmetry is not maintained. This discrepancy arises because electrons are indistinguishable particles, whereas qubits are distinguishable. To correctly emulate the behavior of fermions, several mapping techniques have been developed that preserve the necessary anti-commutation relations.

In this tutorial, we explain how to perform mapping from `OpenFermion`'s `FermionOperator`  to QURI Parts `Operator`, where we provide 3 types of mapping:

1. Jordan-Wigner mapping
2. Bravyi-Kitaev mapping
3. Symmetry-conserving Bravyi-Kitaev mapping

## Prerequisite

QURI Parts modules used in this tutorial: `quri-parts-core`, and `quri-parts-openfermion`. You can install them as follows:


```ipython3
!pip install "quri_parts[openfermion]"
```

## Overview

Here we set up a Fermi-Hubbard hamiltonian for demonstration:


```python
from openfermion import fermi_hubbard

n_site = 2
hamiltonian = fermi_hubbard(x_dimension=n_site, y_dimension=1, tunneling=1, coulomb=2)
print("Fermi-Hubbard Hamiltonian:")
print(hamiltonian)
```
```output
Fermi-Hubbard Hamiltonian:
2.0 [0^ 0 1^ 1] +
-1.0 [0^ 2] +
-1.0 [1^ 3] +
-1.0 [2^ 0] +
2.0 [2^ 2 3^ 3] +
-1.0 [3^ 1]
```
where $i$^ denotes $c_i^{\dagger}$ and $j$ denotes $c_j$ in [・]. For example, [0^ 2] denotes $c_0^{\dagger}c_2$. Note that this is a Hamiltonian written in the form of second quantization.

In QURI Parts, we provide mapping objects that generates:

- `OpenFermion` operator mapper: a function that maps
    - `openfermion.ops.FermionOperator`
    - `openfermion.ops.InteractionOperator`
    - `openfermion.ops.MajoranaOperator`
    
    to QURI Parts `Operator`.

- state mapper: A function that maps the occupation number state:

    $$
    | \Psi \rangle = c_i^{\dagger} c_j^{\dagger} \cdots c_k^{\dagger} | 00\cdots 0\rangle
    $$

    to a `ComputationalBasisState`.
- inverse state mapper: A function that maps a `ComputationalBasisState` to the occupation number state.

We use Jordan-Wigner mapping as an example. The steps of obtaining the mappers in QURI Parts are:

1. Create a mapping object.
2. Retrieve the mappers by accessing corresponding properties.

We first create a mapping object that performs Jordan-Wigner mapping for a system with 4 spin orbitals.


```python
from quri_parts.openfermion.transforms import jordan_wigner
jw_mapping = jordan_wigner(n_spin_orbitals=2*n_site)
```

### Map a Fermion operator to qubit operator


```python
operator_mapper = jw_mapping.of_operator_mapper
qubit_hamiltonian = operator_mapper(hamiltonian)
print(qubit_hamiltonian)
```
```output
(-0.5+0j)*Y0 Z1 Y2 + (-0.5+0j)*X0 Z1 X2 + (-0.5+0j)*Y1 Z2 Y3 + (-0.5+0j)*X1 Z2 X3 + (1+0j)*I + (-0.5+0j)*Z1 + (-0.5+0j)*Z0 + (0.5+0j)*Z0 Z1 + (-0.5+0j)*Z3 + (-0.5+0j)*Z2 + (0.5+0j)*Z2 Z3
```
### Map an occupation state to a `ComputationalBasisState`

Let's look at what the occupation state $|0, 3\rangle = c_0^{\dagger} c_3^{\dagger}|0000\rangle$ is mapped to under Jordan-Wigner mapping.


```python
state_mapper = jw_mapping.state_mapper

occ_state = [0, 3]
cb_state = state_mapper(occ_state)

print("Occupation state:", "|" + ", ".join(map(str, occ_state)) + ">")
print("ComputationalBasisState:", cb_state)
print("State preparation circuit:")
for g in cb_state.circuit.gates:
    print(g)
```
```output
Occupation state: |0, 3>
ComputationalBasisState: ComputationalBasisState(qubit_count=4, bits=0b1001, phase=0π/2)
State preparation circuit:
QuantumGate(name='X', target_indices=(0,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=())
QuantumGate(name='X', target_indices=(3,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=())
```
### Map a `ComputationalBasisState` to an occupation state

We look at what the computational basis state $|1011\rangle$ is mapped to under Jordan-Wigner mapping.


```python
from quri_parts.core.state import quantum_state

inv_state_mapper = jw_mapping.inv_state_mapper

cb_state = quantum_state(n_qubits=2*n_site, bits=0b1011)
occ_state = inv_state_mapper(cb_state)

print("ComputationalBasisState:", cb_state)
print("Occupation state:", "|" + ", ".join(map(str, occ_state)) + ">")
```
```output
ComputationalBasisState: ComputationalBasisState(qubit_count=4, bits=0b1011, phase=0π/2)
Occupation state: |0, 1, 3>
```
The Bravyi-Kitaev mapping and symmetry conserving Bravyi-Kitaev mapping can also be used in a similar way.

## Interface

Here, we introduce the two pillars of performing mappings in QURI Parts: `OpenFermionQubitMapperFactory` and `OpenFermionQubitMapping`.

### `OpenFermionQubitMapping`

An `OpenFermionQubitMapping` is a mapping object that holds the configuration of a state you want to transform or act operators onto. We will refer it as a "mapping object" for short throughout this tutorial. The configuration of a state includes:

- $N_s$: number of spin-orbitals
- $N_e$: number of electrons
- $s_z$: $z$-component of the state's spin

Operator mapper, state mapper and inverse state mapper are retrieved as properties. The `jw_mapping` variable we created in the last section is an `OpenFermionQubitMapping` object. They are generated by `OpenFermionQubitMapperFactory`, which we will introduce in the next subsection.

### `OpenFermionQubitMapperFactory`

An `OpenFermionQubitMapperFactory` is an object for generating `OpenFermionQubitMapping`s. In QURI Parts, we provide the following `OpenFermionQubitMapperFactory`s.


```python
# Jordan-Wigner
from quri_parts.openfermion.transforms import jordan_wigner

# Bravyi-Kitaev
from quri_parts.openfermion.transforms import bravyi_kitaev

# Symmetry conserving Bravyi-Kitaev
from quri_parts.openfermion.transforms import symmetry_conserving_bravyi_kitaev as scbk
```

An `OpenFermionQubitMapperFactory` can also generate mappers with the:

- `get_of_operator_mapper`
- `get_state_mapper`
- `get_inv_state_mapper`

methods by passing in the state configuration. In addition, they hold information about how many qubits are required to perform the mapping for a system of given number of spin orbitals and vice versa. You may obtain the information with:

- `n_qubits_required`: Number of qubits required to perform the mapping for a system of $n$ spin orbitals.
- `n_spin_orbitals`: Number of spin orbtials the system contains if the mapped computational basis state contains $n$ qubits.

## Jordan-Wigner Mapping

We first look at a brief overview of Jordan-Wigner Mapping. Jordan-Wigner Mapping is given by

$$
\begin{align}
    c_j^{\dagger} &= Z_{N_s}\otimes\cdots\otimes Z_{j+1}\otimes\frac{1}{2}(X_j-iY_j)\\
    c_j &= Z_{N_s}\otimes\cdots\otimes Z_{j+1}\otimes\frac{1}{2}(X_j+iY_j)
\end{align}
$$
where $X_i$, $Y_i$ and $Z_i$ are Pauli operator. $X_j+iY_j$ and $X_j-iY_j$ are ladder operators and satisfy 

$$
\begin{align}
    \frac{1}{2}(X_j+iY_j)|1\rangle&=|0\rangle\\
    \frac{1}{2}(X_j-iY_j)|0\rangle&=|1\rangle
\end{align}
$$

$Z_i$ in the equation representing the mapping represent the antisymmetry of the fermionic wavefunction.

Jordan-Wigner mapping can be performed with the `jordan_wigner` object in QURI Parts. We first look at the relation between the number of spin orbitals and the number of qubits when Jordan-Wigner mapping is under consideration.


```python
from quri_parts.openfermion.transforms import jordan_wigner

n_spin_orbitals = 4
n_qubits_required = jordan_wigner.n_qubits_required(n_spin_orbitals=4)
print(
    f"{n_qubits_required} qubits are required to perform Jordan-Wigner mapping for a {n_spin_orbitals}-spin-orbital system."
)

n_qubits = 4
n_spin_orbitals = jordan_wigner.n_spin_orbitals(n_qubits=4)
print(
    f"{n_spin_orbitals} spin orbitals are present in a system if the Jordan-Wigner-mapped "
    f"computational basis state contains {n_qubits} qubits."
)
```
```output
4 qubits are required to perform Jordan-Wigner mapping for a 4-spin-orbital system.
4 spin orbitals are present in a system if the Jordan-Wigner-mapped computational basis state contains 4 qubits.
```
Now, let's generate a Jordan-Wigner mapping object. For Jordan-Wigner mapping, only the number of spin orbitals is required for creating the mapping object. `n_fermions` or `sz` are ignored automatically if they are passed in. Let's create one for a system with 4 spin orbitals. 


```python
n_spin_orbitals = 2 * n_site
jw_mapping = jordan_wigner(n_spin_orbitals)
```

Let's retrieve mappers from it.

### Operator mapper


```python
jw_operator_mapper = jw_mapping.of_operator_mapper
qubit_hamiltonian = jw_operator_mapper(hamiltonian)
print(qubit_hamiltonian)
```
```output
(-0.5+0j)*Y0 Z1 Y2 + (-0.5+0j)*X0 Z1 X2 + (-0.5+0j)*Y1 Z2 Y3 + (-0.5+0j)*X1 Z2 X3 + (1+0j)*I + (-0.5+0j)*Z1 + (-0.5+0j)*Z0 + (0.5+0j)*Z0 Z1 + (-0.5+0j)*Z3 + (-0.5+0j)*Z2 + (0.5+0j)*Z2 Z3
```
### State mapper

Let's look at what the occupation state $|0, 3\rangle = c_0^{\dagger} c_3^{\dagger}|0000\rangle$ is mapped to under Jordan-Wigner mapping.


```python
jw_state_mapper = jw_mapping.state_mapper

occ_state = [0, 3]
cb_state = jw_state_mapper(occ_state)

print("Occupation state:", "|" + ", ".join(map(str, occ_state)) + ">")
print("ComputationalBasisState:", cb_state)
print("State preparation circuit:")
for g in cb_state.circuit.gates:
    print(g)
```
```output
Occupation state: |0, 3>
ComputationalBasisState: ComputationalBasisState(qubit_count=4, bits=0b1001, phase=0π/2)
State preparation circuit:
QuantumGate(name='X', target_indices=(0,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=())
QuantumGate(name='X', target_indices=(3,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=())
```
### Inverse state mapper

We look at what the computational basis state $|1011\rangle$ is mapped to under Jordan-Wigner mapping.


```python
from quri_parts.core.state import quantum_state

jw_inv_state_mapper = jw_mapping.inv_state_mapper

cb_state = quantum_state(n_qubits=2*n_site, bits=0b1011)
occ_state = jw_inv_state_mapper(cb_state)

print("ComputationalBasisState:", cb_state)
print("Occupation state:", "|" + ", ".join(map(str, occ_state)) + ">")
```
```output
ComputationalBasisState: ComputationalBasisState(qubit_count=4, bits=0b1011, phase=0π/2)
Occupation state: |0, 1, 3>
```
### Alternative way of obtaining the mappers

As explained in the [interface](#openfermionqubitmapperfactory) section, mappers can also be generated by the `jordan_wigner` object without creating the mapping object. They can be done with:


```python
jw_operator_mapper = jordan_wigner.get_of_operator_mapper(n_spin_orbitals=2*n_site)
jw_state_mapper = jordan_wigner.get_state_mapper(n_spin_orbitals=2*n_site)
jw_inv_state_mapper = jordan_wigner.get_inv_state_mapper(n_spin_orbitals=2*n_site)
```

## Bravyi-Kitaev Mapping

We first look at a brief overview of Bravyi-Kitaev Mapping. 

In Jordan-Wigner Mapping, the number of occupied spin orbitals can be determined by looking at a single qubit, and parity (whether the sign of the quantum state changes due to antisymmetry of the wavefunction) can be determined by looking at multiple qubits. That is, the number of occupancies is stored locally and parity information is stored non-locally in qubits. Bravyi-Kitaev Mapping is a method to reduce the number of Pauli Z operators needed by making both non-local.

Each qubit of the generated quantum state is obtained by multiplying the Bravyi-Kitaev transformation matrix by the quantum state from the Jordan-Winger transformation. Bravyi-Kitaev transformation matrix is

$$
\beta_{2^x}= \left(
\begin{matrix} 
\beta_{2^{x-1}} & 0/1 \\ 
0 & \beta_{2^{x-1}} 
\end{matrix} 
\right)
$$
where $\beta_1=(1)$ and $0/1$ means that only the top row is lined with 1s, and the other rows are lined with 0s. With this transformation, the odd-numbered qubits conserve the number of occupied spin orbitals corresponding to them, and the even-numbered qubits conserve the parity of the number of occupied spin orbitals in the multiple. We can examine only a certain number of qubits to check the number of occupied spin orbitals and the parity when the creation and annihilation operators are acted upon for a given spin orbital.
 
Bravyi-Kitaev mapping can be performed with the `bravyi_kitaev` object in QURI Parts. We first look at the relation between the number of spin orbitals and the number of qubits when Bravyi-Kitaev mapping is under consideration.


```python
from quri_parts.openfermion.transforms import bravyi_kitaev

n_spin_orbitals = 4
n_qubits_required = bravyi_kitaev.n_qubits_required(n_spin_orbitals=4)
print(
    f"{n_qubits_required} qubits are required to perform Bravyi-Kitaev mapping for a {n_spin_orbitals}-spin-orbital system."
)

n_qubits = 4
n_spin_orbitals = bravyi_kitaev.n_spin_orbitals(n_qubits=4)
print(
    f"{n_spin_orbitals} spin orbitals are present in a system if the Bravyi-Kitaev-mapped "
    f"computational basis state contains {n_qubits} qubits."
)
```
```output
4 qubits are required to perform Bravyi-Kitaev mapping for a 4-spin-orbital system.
4 spin orbitals are present in a system if the Bravyi-Kitaev-mapped computational basis state contains 4 qubits.
```
Now, let's generate a Bravyi-Kitaev mapping object. For Bravyi-Kitaev mapping, only the number of spin orbitals is required for creating the mapping object. `n_fermions` or `sz` are ignored automatically if they are passed in. Let's create one for a system with 4 spin orbitals. 


```python
n_spin_orbitals = 2 * n_site
bk_mapping = bravyi_kitaev(n_spin_orbitals)
```

Let's retrieve mappers from it.

### Operator mapper


```python
bk_operator_mapper = bk_mapping.of_operator_mapper
qubit_hamiltonian = bk_operator_mapper(hamiltonian)
print(qubit_hamiltonian)
```
```output
(-0.5+0j)*X0 Y1 Y2 + (0.5+0j)*Y0 Y1 X2 + (0.5+0j)*Z0 X1 Z3 + (-0.5+0j)*X1 Z2 + (1+0j)*I + (-0.5+0j)*Z0 Z1 + (-0.5+0j)*Z0 + (0.5+0j)*Z1 + (-0.5+0j)*Z1 Z2 Z3 + (-0.5+0j)*Z2 + (0.5+0j)*Z1 Z3
```
### State mapper

Let's look at what the occupation state $|0, 3\rangle = c_0^{\dagger} c_3^{\dagger}|0000\rangle$ is mapped to under Bravyi-Kitaev mapping. When 4 qubits, transformation matrix is

$$
\beta_{4}= \left(
\begin{matrix} 
1&1&1&1\\ 
0&1&0&0 \\
0&0&1&1\\
0&0&0&1
\end{matrix} 
\right)
$$
Therefore after mapping, it takes the form of $(1\,0\,0\,1)^\mathrm{T}$ multiplied by $\beta_{4}$, i.e $(0\, 0\, 1\, 1)$ (note that taking "mod 2").


```python
bk_state_mapper = bk_mapping.state_mapper

occ_state = [0, 3]
cb_state = bk_state_mapper(occ_state)

print("Occupation state:", "|" + ", ".join(map(str, occ_state)) + ">")
print("ComputationalBasisState:", cb_state)
print("State preparation circuit:")
for g in cb_state.circuit.gates:
    print(g)
```
```output
Occupation state: |0, 3>
ComputationalBasisState: ComputationalBasisState(qubit_count=4, bits=0b11, phase=0π/2)
State preparation circuit:
QuantumGate(name='X', target_indices=(0,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=())
QuantumGate(name='X', target_indices=(1,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=())
```
### Inverse state mapper

We look at what the computational basis state $|0011\rangle$ is mapped to under Bravyi-Kitaev mapping.


```python
from quri_parts.core.state import quantum_state

bk_inv_state_mapper = bk_mapping.inv_state_mapper

cb_state = quantum_state(n_qubits=2*n_site, bits=0b11)
occ_state = bk_inv_state_mapper(cb_state)

print("ComputationalBasisState:", cb_state)
print("Occupation state:", "|" + ", ".join(map(str, occ_state)) + ">")
```
```output
ComputationalBasisState: ComputationalBasisState(qubit_count=4, bits=0b11, phase=0π/2)
Occupation state: |0, 3>
```
### Alternative way of obtaining the mappers

As explained in the [interface](#openfermionqubitmapperfactory) section, mappers can also be generated by the `bravyi_kitaev` object without creating the mapping object. They can be done with:


```python
bk_operator_mapper = bravyi_kitaev.get_of_operator_mapper(n_spin_orbitals=2*n_site)
bk_state_mapper = bravyi_kitaev.get_state_mapper(n_spin_orbitals=2*n_site)
bk_inv_state_mapper = bravyi_kitaev.get_inv_state_mapper(n_spin_orbitals=2*n_site)
```

## Symmetry conserving Bravyi-Kitaev Mapping

Symmetry conserving Bravyi-Kitaev (SCBK) mapping can be performed with the `symmetry_conserving_bravyi_kitaev` object in QURI Parts. We first look at the relation between the number of spin orbitals and the number of qubits when SCBK mapping is under consideration.


```python
from quri_parts.openfermion.transforms import symmetry_conserving_bravyi_kitaev as scbk

n_spin_orbitals = 4
n_qubits_required = scbk.n_qubits_required(n_spin_orbitals=n_spin_orbitals)
print(
    f"{n_qubits_required} qubits are required to perform SCBK mapping for a {n_spin_orbitals}-spin-orbital system."
)

n_qubits = 2
n_spin_orbitals = scbk.n_spin_orbitals(n_qubits=n_qubits)
print(
    f"{n_spin_orbitals} spin orbitals are present in a system if the SCBK-mapped "
    f"computational basis state contains {n_qubits} qubits."
)
```
```output
2 qubits are required to perform SCBK mapping for a 4-spin-orbital system.
4 spin orbitals are present in a system if the SCBK-mapped computational basis state contains 2 qubits.
```
Now, let's generate a SCBK mapping object. For SCBK mapping, we need to be particular careful about the configuration of the state. Thus, all `n_spin_orbitals`, `n_fermions` and `sz` are required to create a SCBK mapping object. Let's create one for a system with 4 spin orbitals, 2 electrons and $s_z = 0$.


```python
n_spin_orbitals = 2 * n_site
n_fermions = 2
sz = 0
scbk_mapping = scbk(n_spin_orbitals, n_fermions, sz)
```

Let's retrieve mappers from it.

### Operator mapper


```python
scbk_operator_mapper = scbk_mapping.of_operator_mapper
qubit_hamiltonian = scbk_operator_mapper(hamiltonian)
print(qubit_hamiltonian)
```
```output
-1.0*X0 + -1.0*X1 + 1.0*I + 1.0*Z0 Z1
```
### State mapper

Let's look at what the occupation state $|0, 3\rangle = c_0^{\dagger} c_3^{\dagger}|0000\rangle$ is mapped to under SCBK mapping.


```python
scbk_state_mapper = scbk_mapping.state_mapper

occ_state = [0, 3]
cb_state = scbk_state_mapper(occ_state)

print("Occupation state:", "|" + ", ".join(map(str, occ_state)) + ">")
print("ComputationalBasisState:", cb_state)
print("State preparation circuit:")
for g in cb_state.circuit.gates:
    print(g)
```
```output
Occupation state: |0, 3>
ComputationalBasisState: ComputationalBasisState(qubit_count=2, bits=0b1, phase=0π/2)
State preparation circuit:
QuantumGate(name='X', target_indices=(0,), control_indices=(), classical_indices=(), params=(), pauli_ids=(), unitary_matrix=())
```
### Inverse state mapper

We look at what the computational basis state $|01\rangle$ is mapped to under SCBK mapping.


```python
from quri_parts.core.state import quantum_state

scbk_inv_state_mapper = scbk_mapping.inv_state_mapper

n_spin_orbitals = 2*n_site
n_qubits = scbk.n_qubits_required(n_spin_orbitals)
cb_state = quantum_state(n_qubits=n_qubits, bits=0b01)
occ_state = scbk_inv_state_mapper(cb_state)

print("ComputationalBasisState:", cb_state)
print("Occupation state:", "|" + ", ".join(map(str, occ_state)) + ">")
```
```output
ComputationalBasisState: ComputationalBasisState(qubit_count=2, bits=0b1, phase=0π/2)
Occupation state: |0, 3>
```
### Alternative way of obtaining the mappers

As explained in the [interface](#openfermionqubitmapperfactory) section, mappers can also be generated by the `scbk` object without creating the mapping object. They can be done with:


```python
scbk_operator_mapper = scbk.get_of_operator_mapper(n_spin_orbitals=2*n_site, n_fermions=n_fermions, sz=sz)
scbk_state_mapper = scbk.get_state_mapper(n_spin_orbitals=2*n_site, n_fermions=n_fermions, sz=sz)
scbk_inv_state_mapper = scbk.get_inv_state_mapper(n_spin_orbitals=2*n_site, n_fermions=n_fermions, sz=sz)
```
