# Molecular Hamiltonian Generation

This tutorial is for demonstrating how to obtain the molecular orbital (MO) electron integrals ($h_{ij}$ and $h_{ijkl}$) as well as the molecular Hamiltonian ($H$).

Here, we adopt the physicists' convention for the electron integrals, so that they are related to the Hamiltonian by the equation:

$$
\begin{equation}
H = E_{\text{nuc}} + \sum_{i, j = 1}^{N} h_{ij} c_i^{\dagger} c_j + \frac{1}{2} \sum_{i, j, k, l = 1}^{N} h_{ijkl} c_i^{\dagger} c_j^{\dagger} c_k c_l
\end{equation}
$$

where 

- $E_{\text{nuc}}$ is the nuclear repulsion energy.
- $h_{ij}$ is the 1-electron MO integral (physicist's convention).
- $h_{ijkl}$ is the 2-electron MO integral (physicist's convention).
- $c_i^{\dagger}$, $c_i$ are the fermionic creation and annihilation operators on the i-th _spin_ orbtial.
- $N$ is the number of spin oribtals.

## Overview

We give a brief overview on how to easily generate the Hamiltonian for a water molecule with QURI Parts. The parameters we use are:

- charge: 0
- spin: 0
- [basis](https://en.wikipedia.org/wiki/Basis_set_(chemistry)): sto-3g
- fermion-qubit mapping: Jordan-Wigner

We first generate the hamiltonian where all spatial orbitals are active. This configuration is referred to as the "full space" through out this tutorial.


```python
from pyscf import gto, scf
from quri_parts.pyscf.mol import get_spin_mo_integrals_from_mole
from quri_parts.openfermion.mol import get_qubit_mapped_hamiltonian

mole = gto.M(atom="O 0 0 0; H 0.2774 0.8929 0.2544; H 0.6068, -0.2383, -0.7169")
mf = scf.RHF(mole).run(verbose=0)

hamiltonian, jw_mapping = get_qubit_mapped_hamiltonian(
    *get_spin_mo_integrals_from_mole(mole, mf.mo_coeff)
)
```

We can also compute the active space hamiltonian with active space configuration

- number of active electrons: 6
- number of active orbitals: 4


```python
from quri_parts.chem.mol import cas

cas_hamiltonian, cas_jw_mapping = get_qubit_mapped_hamiltonian(
    *get_spin_mo_integrals_from_mole(mole, mf.mo_coeff, cas(6, 4))
)
```

Note that a mapping object is returned along with the qubit hamiltonian.

## Generate spin MO electron integrals

The molecular hamiltonian is computed based on spin MO electron integrals. They are the $h_{ij}$ and $h_{ijkl}$ coefficients in eq.(1). We will have detailed tutorials explaining how they are computed in QURI Parts and the objects that hold them. For now, we can generate them with the `get_spin_mo_integrals_from_mole` function, which only requires a `pyscf.gto.Mole` object and the corresponding mo coefficients.

Here we show how we can generate the full space and active space electron integrals


```python
full_space, mo_eint_set = get_spin_mo_integrals_from_mole(mole, mf.mo_coeff)
active_space, cas_mo_eint_set = get_spin_mo_integrals_from_mole(mole, mf.mo_coeff, cas(6, 4))
```

`get_spin_mo_integrals_from_mole` returns the active space along with the electron integral The `full_space` variable here is an `ActiveSpace` object with `n_active_electron` being the total number of electrons in the molecule, and `n_active_orbitals` being the total number of spatial orbitals in the molecule.

## Obtaining the fermionic hamiltonian

As the fermionic hamiltonian is directly related to the electron integrals via eq.(1), we can compute the fermionic hamiltonian with them with the `get_fermionic_hamiltonian` function.


```python
from quri_parts.openfermion.mol import get_fermionic_hamiltonian

fermionic_hamiltonian = get_fermionic_hamiltonian(mo_eint_set)
cas_fermionic_hamiltonian = get_fermionic_hamiltonian(cas_mo_eint_set)
```

## Mapping the fermionic Hamiltonian to qubit hamiltonian

The fermionic hamiltonian is represented by a `openfermion.InteractionOperator`, where we can use an operator mapper to map it to QURI Parts `Operator`. There are 2 ways this can be done:

1. Construct a mapping object by hand and use it to convert the `FermionicOperator` to `Operator`.
2. Use the `operator_from_of_fermionic_op` function.

We first show method 1.


```python
from quri_parts.openfermion.transforms import jordan_wigner

jw_mapping = jordan_wigner(2 * full_space.n_active_orb, full_space.n_active_ele)
qubit_hamiltonian = jw_mapping.of_operator_mapper(fermionic_hamiltonian)

cas_jw_mapping = jordan_wigner(2 * active_space.n_active_orb, active_space.n_active_ele)
cas_qubit_hamiltonian = cas_jw_mapping.of_operator_mapper(cas_fermionic_hamiltonian)
```

This computes the correct qubit Hamiltonian, but is a bit cumbersome because the mapping object is computed by hand. The `operator_from_of_fermionic_op` function bypasses this shortcoming by taking in the active space and generates the mapping object automatically.


```python
from quri_parts.openfermion.mol import operator_from_of_fermionic_op

qubit_hamiltonian, jw_mapping = operator_from_of_fermionic_op(
    fermionic_hamiltonian,
    full_space,
    sz = None,  # default to None
    fermion_qubit_mapping=jordan_wigner  # default to jordan_wigner
)
cas_qubit_hamiltonian, cas_jw_mapping = operator_from_of_fermionic_op(
    cas_fermionic_hamiltonian,
    active_space,
    sz = None,  # default to None
    fermion_qubit_mapping=jordan_wigner  # default to jordan_wigner
)
```

## Qubit hamiltonian from MO integral

Finally, we introduce the `get_qubit_mapped_hamiltonian` function demonstrated in the overview section. It serves as a shortcut that completely bypasses the fermionic hamiltonian


```python
qubit_hamiltonian, jw_mapping = get_qubit_mapped_hamiltonian(
    full_space,
    mo_eint_set,
    sz = None,  # default to None
    fermion_qubit_mapping=jordan_wigner  # default to jordan_wigner
)

cas_qubit_hamiltonian, cas_jw_mapping = get_qubit_mapped_hamiltonian(
    active_space,
    cas_mo_eint_set,
    sz = None,  # default to None
    fermion_qubit_mapping=jordan_wigner  # default to jordan_wigner
)
```
