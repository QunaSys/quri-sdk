# Operators

In this tutorial, we introduce 2 objects, `Operator` and `PauliLabel`, that represents operators in quantum mechanics. You may construct various physical observables with them. In QURI Parts, we mainly work with operators consists of Pauli strings.

## PauliLabel

Pauli strings are ubiquitous in quantum computation. In QURI Parts, they are represented by `PauliLabel`. This section is devoted to explain what `PauliLabel` is made of and how to create them. We first start with their basic building block: Pauli matrices.

### Pauli matrices

In QURI Parts, Pauli matrices are represented by an `Enum`: `SinglePauli`. They are not objects to be used for any computations directly, they are simply labels of what Pauli matrices a `PauliLabel` hold.


```python
from quri_parts.core.operator import SinglePauli

assert SinglePauli.X == 1
assert SinglePauli.Y == 2
assert SinglePauli.Z == 3
```

### Pauli strings

As mentioned previously, Pauli strings are represented by `PauliLabel`. We introduce the interface and how to create one.

#### Interface

In QURI Parts, a `PauliLabel` represents a Pauli string. It is a `frozenset` of `tuple[qubit_index, SinglePauli]`.

```python
class PauliLabel(frozenset(tuple[int, int])):
    """First int represents the qubit index and
    the second represents a `SinglePauli`
    """
    ...
```

#### Creating a `PauliLabel`

There are various ways of creating a `PauliLabel`. Here, we introduce the simplest one, which is using the `pauli_label` function. For basic usage, the `pauli_label` function accepts 2 types of inputs:
1. A `str` that looks like a Pauli string.
2. Sequence of (qubit index, `SinglePauli`) pairs.

##### Create with a `str`

We can create a `PauliLabel` by passing in a `str` that looks like a human-readable Pauli string


```python
from quri_parts.core.operator import PauliLabel, pauli_label

print("Create without spacing:", pauli_label("X0 Y1 Z2 Z3 X4 Y5"))
print("Create with spacing:   ", pauli_label("X 0  Y 1  Z 2  Z 3  X 4  Y 5"))
```
```output
Create without spacing: X0 Y1 Z2 Z3 X4 Y5
Create with spacing:    X0 Y1 Z2 Z3 X4 Y5
```
Note that the order of Pauli matrices does not matter.


```python
print("Create with X0 Y1 Z2:", pauli_label("X0 Y1 Z2"))
print("Create with X0 Z2 Y1:", pauli_label("X0 Z2 Y1"))
print(pauli_label("X0 Y1 Z2") == pauli_label("X0 Z2 Y1"))
```
```output
Create with X0 Y1 Z2: X0 Y1 Z2
Create with X0 Z2 Y1: X0 Y1 Z2
True
```
##### Create with a sequence of `(qubit_index, SinglePauli)`

We can also create a `PauliLabel` by passing a sequence of `(qubit_index, SinglePauli)` into the `pauli_label` function.


```python
print(pauli_label([(0, SinglePauli.X), (1, SinglePauli.Z)]))
print(pauli_label(zip((0, 1), (SinglePauli.X, SinglePauli.Z))))
```
```output
X0 Z1
X0 Z1
```
There is a special `PauliLabel`: `PAULI_IDENTITY`. It represents the identity operator and is a `PauliLabel` with no entry.


```python
from quri_parts.core.operator import PAULI_IDENTITY

# PauliLabel() represents an empty `frozenset`.
print(PauliLabel() == PAULI_IDENTITY)
```
```output
True
```
#### Methods `PauliLabel` provides

The `PauliLabel` provides several methods that provides information about itself.

- `index_and_pauli_id_list`: A property that returns a tuple of (list[qubit index], list[`SinglePauli`])


```python
pauli_label("X0 Y1 Z2").index_and_pauli_id_list
```



```output
([0, 1, 2], [<SinglePauli.X: 1>, <SinglePauli.Y: 2>, <SinglePauli.Z: 3>])
```


- `qubit_indices`: The list of qubits this `PauliLabel` acts on.


```python
pauli_label("X0 Y1 Z2").qubit_indices()
```



```output
[0, 1, 2]
```


- `pauli_at`: The Pauli matrix at the specified qubit. If the operator at the specified qubit index is identity, it returns `None`.


```python
print(pauli_label("X0 Y1 Z2").pauli_at(0))
print(pauli_label("X0 Y1 Z2").pauli_at(1))
print(pauli_label("X0 Y1 Z2").pauli_at(2))
print(pauli_label("X0 Y1 Z2").pauli_at(3))
```
```output
1
2
3
None
```
## Operator

Here, we introduce the `Operator` object. The `Operator` object represents a complex linear combination of `PauliLabel`s.

### Interface
In QURI Parts, it is implmented as a dictionary with `PauliLabel` as key and complex number as value. So, you can create an `Operator` with a dictionary.


```python
from quri_parts.core.operator import Operator

op = Operator(
    {
        PAULI_IDENTITY: 8 + 1j,
        pauli_label("X0 Y2"): -3
    }
)

print(op)
```
```output
(8+1j)*I + -3*X0 Y2
```
### Arithmetics with `Operator`

You can add terms to an `Operator` with the `add_term` method, which updates the `Operator` in place. If a `PauliLabel` already exists in the `Operator`, it updates the coeffcient. Suppose the new coefficient is 0, the `PauliLabel` will be dropped from the `Operator`.


```python
op = Operator(
    {
        PAULI_IDENTITY: 8 + 1j,
        pauli_label("X0 Y2"): -3
    }
)

print(op, "\n")

# Add a new term to the Operator
pl, coeff = pauli_label("Y0 Z3"), 10+1j
print(f"Add {coeff} * {pl}:")
op.add_term(pl, coeff)
print(op, "\n")

# Add a `PauliLabel` that already exists in `Operator` to update the coefficient
pl, coeff = PAULI_IDENTITY, -6
print(f"Add {coeff} * {pl}:")
op.add_term(pl, coeff)
print(op, "\n")

# Add a `PauliLabel` that already exists in `Operator` to cancel the term
pl, coeff = pauli_label("X0 Y2"), 3
print(f"Add {coeff} * {pl}:")
op.add_term(pl, coeff)
print(op)
```
```output
(8+1j)*I + -3*X0 Y2 

Add (10+1j) * Y0 Z3:
(8+1j)*I + -3*X0 Y2 + (10+1j)*Y0 Z3 

Add -6 * I:
(2+1j)*I + -3*X0 Y2 + (10+1j)*Y0 Z3 

Add 3 * X0 Y2:
(2+1j)*I + (10+1j)*Y0 Z3
```
There are also 2 properties provided:
- n_terms: The number of terms in the `Operator`.
- constant: Returns the coefficient of `PAULI_IDENTITY` in the `Operator`. It gives 0 if `PAULI_IDENTITY` is not present.


```python
op = Operator(
    {
        PAULI_IDENTITY: 8 + 1j,
        pauli_label("X0 Y2"): -3
    }
)


print("n_terms:", op.n_terms)
print("constant:", op.constant)
```
```output
n_terms: 2
constant: (8+1j)
```
The `Operator` object also provides several methods for basic arithmetics:


```python
op1 =  Operator({pauli_label("X0 Z1"): 8j})
op2 =  Operator({pauli_label("Y1"): -4})

print("op1 = ", op1)
print("op2 = ", op2)

# Addition
print("")
print("Addition:")
print("op1 + op2", "=", op1 + op2)

# Subtraction
print("")
print("Subtraction:")
print("op1 - op2", "=", op1 - op2)

# Scalar Multiplication
print("")
print("Scalar Multiplication:")
print("op1 * 3j", "=", op1 * 3j)

# Scalar Division
print("")
print("Scalar Division:")
print("op1 / 2j", "=", op1 / 2j)

# Operator Multiplication
print("")
print("Operator Multiplication:")
print("op1 * op2", "=", op1 * op2)
print("op2 * op1", "=", op2 * op1)

# Hermitian conjugation
print("")
print("Hermition Conjgation:")
print("op1^†", "=", op1.hermitian_conjugated())
print("op2^†", "=", op2.hermitian_conjugated())
```
```output
op1 =  8j*X0 Z1
op2 =  -4*Y1

Addition:
op1 + op2 = 8j*X0 Z1 + -4*Y1

Subtraction:
op1 - op2 = 8j*X0 Z1 + 4*Y1

Scalar Multiplication:
op1 * 3j = (-24+0j)*X0 Z1

Scalar Division:
op1 / 2j = (4+0j)*X0 Z1

Operator Multiplication:
op1 * op2 = (-32+0j)*X0 X1
op2 * op1 = (32+0j)*X0 X1

Hermition Conjgation:
op1^† = -8j*X0 Z1
op2^† = -4*Y1
```
There is also a special `Operator`: `zero()` that represents a zero operator. It is an `Operator` created with an empty dictionary.


```python
from quri_parts.core.operator import zero

zero_operator = zero()
print(zero_operator == Operator())
```
```output
True
```
## Helper functions

There are also various other helper functions that provides several arithmetical functionalities for `Operator`. Here we introduce:
- `is_hermitian`
- `commutator`
- `truncate`
- `is_ops_close`
- `get_sparse_matrix`

We provide examples for them below:

### `is_hermitian`

`is_hermitian` checks if an `Operator` is hermitian or not.


```python
from quri_parts.core.operator import is_hermitian

op = Operator({pauli_label("X0"): 1})
print(f"{op} is hermitian:", is_hermitian(op))

op = Operator({pauli_label("X0"): 1j})
print(f"{op} is hermitian:", is_hermitian(op))
```
```output
1*X0 is hermitian: True
1j*X0 is hermitian: False
```
### `commutator`

`commutator` computes the commutator of two operators.


```python
from quri_parts.core.operator import commutator

op1 = Operator({pauli_label("X0"): 1})
op2 = Operator({pauli_label("Y0"): 1})
print(f"[{op1}, {op2}]", "=", commutator(op1, op2))
```
```output
[1*X0, 1*Y0] = 2j*Z0
```
### `truncate`

`truncate` removes `PauliLabel` from `Operator`s if the corresponding coefficient is too small. (Default tolerance is 1e-8.)


```python
from quri_parts.core.operator import truncate
op = Operator(
    {
        pauli_label("Z0 Y1"): 1e-6,
        pauli_label("X0 Y1"): 1e-10,
        pauli_label("X0 Y2"): 1,
    }
)

print("original operator:\n", op)
print("truncated operator:\n", truncate(op))
print("truncated operator with tolerance = 1e-5:\n", truncate(op, 1e-5))
```
```output
original operator:
 1e-06*Z0 Y1 + 1e-10*X0 Y1 + 1*X0 Y2
truncated operator:
 1e-06*Z0 Y1 + 1*X0 Y2
truncated operator with tolerance = 1e-5:
 1*X0 Y2
```
### `is_ops_close`

`is_ops_close` checks if two operators are close up to some tolerance. For the `Operator`s entered in the first and second arguments, this returns whether they are equal within an acceptable margin of error.


```python
from quri_parts.core.operator import is_ops_close
op1 = Operator({pauli_label("X0 Y2"): 1})
op2 = Operator(
    {
        pauli_label("Z0 Y1"): 1e-6,
        pauli_label("X0 Y1"): 1e-10,
        pauli_label("X0 Y2"): 1,
    }
)

print("op1", "=", op1)
print("op2", "=", op2)
print(f"op1 is close to op2 (atol=0):", is_ops_close(op1, op2))
print(f"op1 is close to op2 (atol=1e-5):", is_ops_close(op1, op2, atol=1e-5))
print(f"op1 is close to op2 (atol=1e-8):", is_ops_close(op1, op2, atol=1e-8))
```
```output
op1 = 1*X0 Y2
op2 = 1e-06*Z0 Y1 + 1e-10*X0 Y1 + 1*X0 Y2
op1 is close to op2 (atol=0): False
op1 is close to op2 (atol=1e-5): True
op1 is close to op2 (atol=1e-8): False
```
### `get_sparse_matrix`

`get_sparse_matrix` converts `Operator`s and `PauliLabel`s to a`scipy.sparse.spmatrix`.


```python
from quri_parts.core.operator import get_sparse_matrix
op = Operator({
    PAULI_IDENTITY: -8j,
    pauli_label("X0 Y1"): 1
})

print(get_sparse_matrix(op))
```
```output
<Compressed Sparse Column sparse matrix of dtype 'complex128'
	with 8 stored elements and shape (4, 4)>
  Coords	Values
  (0, 0)	-8j
  (3, 0)	1j
  (1, 1)	-8j
  (2, 1)	1j
  (1, 2)	-1j
  (2, 2)	-8j
  (0, 3)	-1j
  (3, 3)	-8j
```
We can also get the explicit matrix representation


```python
get_sparse_matrix(op).toarray()
```



```output
array([[0.-8.j, 0.+0.j, 0.+0.j, 0.-1.j],
       [0.+0.j, 0.-8.j, 0.-1.j, 0.+0.j],
       [0.+0.j, 0.+1.j, 0.-8.j, 0.+0.j],
       [0.+1.j, 0.+0.j, 0.+0.j, 0.-8.j]])
```


It is often the case that the largest qubit containing a non-trivial Pauli matrix in an `Operator` or a `PauliLabel` is smaller than the number of qubits of the state it acts on. In this case, we can set the qubit count of the state the operator acts on with the `n_qubit` option.


```python
get_sparse_matrix(op, n_qubits=3).toarray()
```



```output
array([[0.-8.j, 0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.-8.j, 0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+1.j, 0.-8.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+1.j, 0.+0.j, 0.+0.j, 0.-8.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.-8.j, 0.+0.j, 0.+0.j, 0.-1.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.-8.j, 0.-1.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j, 0.-8.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j, 0.-8.j]])
```

