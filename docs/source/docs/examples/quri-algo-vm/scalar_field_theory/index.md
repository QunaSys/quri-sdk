# Simulating the $\Phi^4$ Theory on a Quantum Computer

In [1] and [2], algorithms for simulating lattice $\Phi^4$ scalar field lattice on a quantum computer was introduced. This can serve as a first example of simulating general lattice field theory on quantum computers.


The Hamiltonian under consideration is given by:

$$H(\Phi, \Pi) = \frac{1}{2}\Pi^2 + \frac{1}{2}m^2 \Phi^2 + \frac{1}{2}(\nabla \Phi)^2 + \frac{\lambda}{4!}\Phi^4 + J \Phi.$$

Here $\Phi$ and $\Pi$ satisfy the commutation relation:
$$[\Phi(x), \Pi(y)] = i\delta(x - y)$$

To simulate it on a quantum computer, we need to discretize it so that the lattice Hamiltonian takes the form:

$$H_{lat} = a^{D}\sum_{j=1}^N\left(\frac{1}{2}\Pi_j^2 + \frac{1}{2}m^2 \Phi_j^2 + \frac{1}{2a^2}\sum_{e=1}^{D}(\Phi_{j+e} - \Phi_{j})^2 + \frac{\lambda}{4!}\Phi_j^4 + J \Phi_j\right)$$

The lattice field operators satisfy:

$$[\Phi_i, \Pi_j] = ia^{-D}\delta_{i,j}$$

Rescaling with $\Phi_i \rightarrow a^{\frac{-D+1}{2}} \Phi_i $ and $\Pi_i \rightarrow a^{\frac{-D-1}{2}} \Pi_i $ makes the field and the conjugate momentum dimensionless.


```python
def get_qubit_label(in_site_label: int, site_label: int, n_phi_qubit: int) -> int:
    """Converts a local qubit index to the full system's qubit index.
    in_site_label: The qubit label on the local site.
    site_label: The label of the site.
    n_phi_qubit: Number of qubits assigned to each site.
    """
    return n_phi_qubit * site_label + in_site_label
```

In QURI Algo, we can define a subclass of `Problem` containing the parameters that characterize the system. So, we define `DiscreteScalarField1D` that represents the discrete scalar field Hamiltonian in 1 spatial dimension.


```python
from quri_algo.problem import QubitHamiltonian
from dataclasses import dataclass, field
import numpy as np

@dataclass
class DiscreteScalarField1D(QubitHamiltonian):
    """Represents the Hamiltonian of the 1D scalar field:

    Hamiltonian given by:
        H = 1/2 Π_j^2 + 1/2 mb^2 Φ_j^2 + 1/2 (Φ_j - Φ_{j+1})^2 + λ/4! Φ_j^4 + J_j Φ.

    Note:
        1D here means 1 spatial dimension.

    Args:
        n_state_qubit
        n_discretize: Number of points discretizing the field.
        n_phi_qubit: Number of qubits per site.
        mb: Boson mass.
        lam: coupling costant of the phi^4 term.
        external_field: External field strength J.
    """
    n_state_qubit: int = field(init=False)
    n_discretize: int
    n_phi_qubit: int
    mb: float
    lam: float
    external_field: float = 0.0

    def __post_init__(self) -> None:
        self.n_state_qubit = self.n_discretize * self.n_phi_qubit

    @property
    def n_phi_dimension(self) -> int:
        return 2**self.n_phi_qubit

    @property
    def delta_phi(self) -> float:
        return np.sqrt(2 * np.pi * self.mb / self.n_phi_dimension)
```

## Discrete scalar field

The discrete scalar field is designed to satisfy the quantization condition:

$$\Phi_j |\alpha_j \rangle = -\Delta_{\phi} \left(\alpha_j - \frac{N - 1}{2}\right)|\varphi_{\alpha_j} \rangle, \; \alpha_j = 0 \cdots N-1, $$

where $N = 2^n$ with $n$ being the number of qubits assigned to site $j$. The site index $j$ will be suppressed from now and we adopt the conventions:

$$
\begin{equation}
| \varphi_{\alpha_j} \rangle = | \alpha_{n-1} \cdots \alpha_{0} \rangle, \; \alpha = \sum_{q=0}^{n-1} \alpha_q 2^q.
\end{equation}
$$

The discrete field operator on the site $j$ can be expressed as:

$$\Phi = -\Delta_{\phi} \sum_{q=0}^{n-1}  2^{q-1} Z_q, \; \Delta_{\phi} = \sqrt{\frac{2\pi m_b}{N}}$$

which can be implemented as a QURI Parts `Operator`.


```python
from quri_parts.core.operator import Operator, pauli_label

def get_scalar_field_operator(
    site_label: int, n_phi_qubit: int, mass: float
) -> Operator:
    phi = Operator({})
    delta_phi = np.sqrt(2 * np.pi * mass / 2**n_phi_qubit)
    for q in range(n_phi_qubit):
        coeff = - delta_phi * 2**(q) / 2
        l = get_qubit_label(q, site_label, n_phi_qubit)
        phi.add_term(pauli_label(f"Z {l}"), coeff)
    return phi
```

### Example: Check that the operator satisfies the field operator quantization condition.


```python
from quri_parts.core.state import quantum_state
from quri_parts.qulacs.estimator import create_qulacs_general_vector_estimator

site_label = 0
n_qubits = 4
mb = 1
estimator = create_qulacs_general_vector_estimator()

field_op = get_scalar_field_operator(site_label, n_qubits, mb)
b = 0b0101
estimator(field_op, quantum_state(n_qubits, bits=b)), (b - (2**n_qubits - 1)/2) * np.sqrt(2 * np.pi * mb / 2**n_qubits)
```



```output
(_Estimate(value=(-1.5666426716443753+0j), error=0.0), -1.566642671644375)
```


## Discrete conjugate momentum

The discrete conjugate momentum is defined as 

$$
\begin{equation}
\Pi_j = m_b F_j \Phi_j F_j^{\dagger}
\end{equation}
$$
The $F_{j}$ terms is defined as ($j$ is suppressed.)

$$
F = \frac{1}{\sqrt{N}} \sum_{\alpha, \beta = 0}^{N-1} \exp\left[\frac{2\pi i}{N}(\alpha - \frac{N-1}{2})(\beta - \frac{N-1}{2})\right]|\beta\rangle\langle \alpha|
$$

The eigenstate of the conjugate momentum operator is defined as:

$$
|\kappa_\beta\rangle = F|\varphi_\beta\rangle,
$$

where $|\varphi_\beta\rangle$ is the eigenstate of the field operator $\Phi_j$ so that


$$
\Pi|\kappa_\beta\rangle = \left(\beta - \frac{N-1}{2}\right)|\kappa_\beta\rangle,
$$

The transformation $F$ is harder to be represented as a qubit operator. Instead, it can be represented as 

$$F = \left[\bigotimes_{p=0}^{n-1}R_Z(-2^p\delta)\right] \text{QFT} \left[\bigotimes_{q=0}^{n-1}R_Z(-2^q\delta)\right]$$

with $\delta \equiv \frac{(N-1)\pi}{N}$ and QFT corresponds to the quantum Fourier transform:

$$
\text{QFT}|\alpha \rangle = \frac{1}{\sqrt{N}}\sum_{\beta=0}^{N-1} \exp\left(\frac{2\pi i}{N}\alpha \beta \right)|\beta\rangle.
$$

### Quantum Fourier Transform

The quantum Fourier transform can be performed by the following circuit
![qft](qft_circuit.png)

where 

$$
U(k) = 
\begin{pmatrix}
1 & 0 \\
0 & \exp \left(\frac{2\pi i}{2^k}\right)
\end{pmatrix}
$$

and the controlled U gate can be decomposed into CNOTs and U gates as:

$$
CU_{c,t}(\theta) = U_{c}\left(\theta/2\right) CX_{c, t} \; U_{1, t}\left(-\theta/2\right) CX_{c, t} \; U_{1, t}\left(\theta/2\right).
$$

Here $c$ denotes the controlled qubit and $t$ denotes the target qubit. This can be implemented in QURI Parts as:


```python
from quri_parts.circuit import QuantumCircuit, NonParametricQuantumCircuit, ImmutableBoundParametricQuantumCircuit
import numpy as np

def add_controlled_U1_gate(
    circuit: QuantumCircuit, control: int, target: int, angle: float
) -> None:
    circuit.add_U1_gate(control, angle/2)
    circuit.add_CNOT_gate(control, target)
    circuit.add_U1_gate(target, -angle/2)
    circuit.add_CNOT_gate(control, target)
    circuit.add_U1_gate(target, angle/2)
```

Then the QFT circuit can be implemented with:


```python
def create_qft_gate(qubit_count: int) -> ImmutableBoundParametricQuantumCircuit:
    circuit = QuantumCircuit(qubit_count)
    for i in range(qubit_count//2):
        circuit.add_SWAP_gate(i, qubit_count-i-1)

    for target in range(qubit_count):
        circuit.add_H_gate(target)
        for l, control in enumerate(range(target+1, qubit_count)):
            angle = 2 * np.pi/2**(l+2)
            add_controlled_U1_gate(circuit, control, target, angle)

    return circuit.freeze()
```

#### Example: execute the quantum Fourier transform


```python
from quri_parts.qulacs.simulator import evaluate_state_to_vector

n_qubits = 4
n = 2**n_qubits
QFT = create_qft_gate(n_qubits)

for b in range(n):
    qft_state = quantum_state(n_qubits, circuit=QFT, bits=b)
    qft_state_vector = evaluate_state_to_vector(qft_state).vector

    expected_qft_vector = np.array([np.exp(2j * np.pi / n * i * b) for i in range(n)]) / np.sqrt(n)

    assert np.allclose(qft_state_vector, expected_qft_vector)
```

### The linear rotation term

Next, implement the linear rotation terms: $\bigotimes_{p=0}^{n-1}R_Z(-2^p\delta)$.


```python
def create_linear_rotation(qubit_count: int) -> ImmutableBoundParametricQuantumCircuit:
    circuit = QuantumCircuit(qubit_count)
    n = 2**qubit_count
    delta = (n - 1) * np.pi / n

    for q in range(qubit_count):
        angle = - 2**q * delta
        circuit.add_RZ_gate(q, angle)

    return circuit.freeze()
```

#### Example: the linear rotation term


```python
import itertools
n_qubits = 4
n = 2**n_qubits
linear_rotation = create_linear_rotation(n_qubits)

delta = (n - 1) / n * np.pi


for b1, b2 in itertools.product(range(n), repeat=2):

    linear_rotation_state_1 = quantum_state(n_qubits, circuit=linear_rotation, bits=b1)
    linear_rotation_vector_1 = evaluate_state_to_vector(linear_rotation_state_1).vector
    linear_rotation_state_2 = quantum_state(n_qubits, circuit=linear_rotation, bits=b2)
    linear_rotation_vector_2 = evaluate_state_to_vector(linear_rotation_state_2).vector

    n1 = linear_rotation_vector_1[linear_rotation_vector_1.nonzero()]
    n2 = linear_rotation_vector_2[linear_rotation_vector_2.nonzero()]

    assert np.isclose(
        n1/n2, np.exp(-1j * delta * (b1 - b2))
    )
```

### The $F$ operator

Finally, we can combine the functions in the last 2 subsections and finally implement the full circuit for the $F$ operation.


```python
def create_F_gate(qubit_count: int) -> NonParametricQuantumCircuit:
    circuit = QuantumCircuit(qubit_count)
    qft = create_qft_gate(qubit_count)
    linear_rotation = create_linear_rotation(qubit_count)

    circuit.extend(linear_rotation)
    circuit.extend(qft)
    circuit.extend(linear_rotation)

    return circuit
```

#### Example: full $F$ operator


```python
from quri_parts.qulacs.simulator import evaluate_state_to_vector

n_qubits = 4
N = 2**n_qubits
const = (N - 1)/2

F = create_F_gate(n_qubits)
b = 0b1001
conj_eig_state = quantum_state(n_qubits, circuit=F, bits=b)
conj_eig_state_vector = evaluate_state_to_vector(conj_eig_state).vector

expected = np.zeros(2**n_qubits, dtype=np.complex128)
for i in range(N):
    expected[i] = np.exp(
        2j * np.pi / N * (b - const) * (i - const)
    ) / np.sqrt(N)

expected / conj_eig_state_vector    # Only off by overall phase
```



```output
array([-0.99518473+0.09801714j, -0.99518473+0.09801714j,
       -0.99518473+0.09801714j, -0.99518473+0.09801714j,
       -0.99518473+0.09801714j, -0.99518473+0.09801714j,
       -0.99518473+0.09801714j, -0.99518473+0.09801714j,
       -0.99518473+0.09801714j, -0.99518473+0.09801714j,
       -0.99518473+0.09801714j, -0.99518473+0.09801714j,
       -0.99518473+0.09801714j, -0.99518473+0.09801714j,
       -0.99518473+0.09801714j, -0.99518473+0.09801714j])
```


## The Exact Hamiltonian

Here we build the functions for generating the matrix representations of $\Phi_j$, $\Pi_j$, and the hamiltonian $H$ and the local hamiltonian $H_{local, j}$. These will be used extensively in later sections.


```python
import math
import functools
import numpy.typing as npt
from quri_parts.core.operator import get_sparse_matrix


def get_full_matrix(
    n_sites: int, phi_qubit: int, on_site_matrix: npt.NDArray[np.complex128], site_idx: int
) -> npt.NDArray[np.complex128]:
    """Expand an operator on a local site to a matrix of the whole system.
    """
    full = [np.eye(2 ** phi_qubit, dtype=np.complex128) for _ in range(n_sites)]
    full[n_sites - 1 - site_idx] = on_site_matrix
    return functools.reduce(lambda a, b: np.kron(a, b),  full)


def get_F_matrix(n_qubits: int) -> npt.NDArray[np.complex128]:
    """The matrix representation of the F operator on a single site.
    """
    dim = 2**n_qubits
    f_matrix = np.zeros((dim, dim), dtype=np.complex128)
    for a, b in itertools.product(range(dim), repeat=2):
        f_matrix[a, b] = np.exp(2j * np.pi/dim * (a - (dim - 1)/2) * (b - (dim - 1)/2))
    return f_matrix/np.sqrt(dim)


def get_phi_matrix(n_qubits: int, mass: float) -> npt.NDArray[np.complex128]:
    """The matrix representation of the field operator on a single site.
    """
    op = get_scalar_field_operator(0, n_qubits, mass)
    return get_sparse_matrix(op, n_qubits).toarray()


def get_hamiltonian_matrix(system: DiscreteScalarField1D) -> npt.NDArray[np.complex128]:
    """Matrix representation of full Hamiltonian.
    """
    n_qubits = system.n_phi_qubit
    n_site = system.n_discretize
    mb = system.mb
    J = system.external_field
    lam = system.lam
    F_matrix = get_F_matrix(n_qubits)
    phi = get_phi_matrix(n_qubits, mb)

    full_dim = 2**system.n_state_qubit
    hamiltonian = np.zeros((full_dim, full_dim), dtype=np.complex128)

    _phi_cache = {
        site: get_full_matrix(n_site, n_qubits, phi, site) for site in range(n_site)
    }
    for site in range(n_site):
        full_phi = _phi_cache[site]
        full_F = get_full_matrix(n_site, n_qubits, F_matrix, site)
        # conjugate momentum: mb^2 F Φ_j^2 F^{\dagger}
        hamiltonian += 0.5 * mb**2 * (full_F @ full_phi @ full_phi @ full_F.conj().T)

        # J Φ_j
        hamiltonian += J * full_phi

        # 1/2 mb^2 Φ_j^2
        hamiltonian += 0.5 * mb**2 * (full_phi @ full_phi)

        # lam/4! * Φ_j^4
        hamiltonian += (lam/math.factorial(4)) * (full_phi @ full_phi @ full_phi @ full_phi)

        # kin: 1/2 * (Φ_j - Φ_{j+1})^2
        if site < n_site - 1:
            kin_root = _phi_cache[site] - _phi_cache[site + 1]
            hamiltonian += 0.5 * kin_root @ kin_root
    return hamiltonian


def get_local_hamiltonian_matrix(system: DiscreteScalarField1D) -> npt.NDArray[np.complex128]:
    """Matrix representation of local Hamiltonian (on the zeroth site).
    """
    n_qubits = system.n_phi_qubit
    mb = system.mb
    J = system.external_field
    lam = system.lam
    F_matrix = get_F_matrix(n_qubits)
    phi = get_phi_matrix(n_qubits, mb)

    dim = 2**n_qubits
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)

    # conjugate momentum: mb^2 F Φ_j^2 F^{\dagger}
    hamiltonian += 0.5 * mb**2 * (F_matrix @ phi @ phi @ F_matrix.conj().T)

    # J Φ_j
    hamiltonian += J * phi

    # 1/2 mb^2 Φ_j^2
    hamiltonian += 0.5 * mb**2 * (phi @ phi)

    # lam/4! * Φ_j^4
    hamiltonian += (lam/math.factorial(4)) * (phi @ phi @ phi @ phi)

    return hamiltonian

```

## Local state preparation

As a demonstration, we prepare the ground state of the local Hamiltonian on site $j$:

$$H_{local, j} = \frac{1}{2}\Pi^2_{j} + \frac{1}{2}m^2 \Phi^2_j + \frac{\lambda}{4!}\Phi^4_j + J \Phi_j$$

with VQE. We assume the ansatz $|\psi(\vec{\theta})\rangle$ to be the Hardware Efficient Ansatz . The full wave function can then be prepared by:

$$|\Psi\rangle = \bigotimes_{j=0}^{N_{\text{site} -1}} |\psi_j(\vec{\theta}^*)\rangle$$


```python
import math
from quri_parts.algo.ansatz import HardwareEfficient
from quri_parts.algo.optimizer import CostFunction, Params
from quri_parts.qulacs.estimator import create_qulacs_general_vector_estimator
from quri_parts.core.state import ParametricCircuitQuantumState
from quri_parts.circuit import inverse_circuit

estimator = create_qulacs_general_vector_estimator()

def get_cost_function(system: DiscreteScalarField1D, n_layers: int) -> tuple[CostFunction, ParametricCircuitQuantumState]:

    n_local_qubit = system.n_phi_qubit

    F = create_F_gate(n_local_qubit)
    ansatz = HardwareEfficient(n_local_qubit, n_layers)
    ansatz_state = ParametricCircuitQuantumState(n_local_qubit, circuit=ansatz)

    scalar_field_op = get_scalar_field_operator(0, n_local_qubit, system.mb)

    def cost_function(param: Params) -> float:
        bound_state = ansatz_state.bind_parameters(param)
        transformed_state = bound_state.with_gates_applied(inverse_circuit(F))
        mb = system.mb
        lam = system.lam
        J = system.external_field

        # Π^2 = m^2 F Φ^2 F†
        # <Ψ| Π^2 |Ψ> = m^2 <Ψ|F Φ^2 F†|Ψ>
        pi_sq = estimator(0.5 * mb**2 * scalar_field_op * scalar_field_op, transformed_state)

        # 1/2 m^2 Φ^2 + lambda/4! Φ^4 + J Φ
        potential_op = 0.5 * mb**2 * scalar_field_op * scalar_field_op
        potential_op += (lam / math.factorial(4)) * scalar_field_op * scalar_field_op * scalar_field_op * scalar_field_op
        potential_op += J * scalar_field_op
        non_dyn = estimator(potential_op, bound_state)
        cost = pi_sq.value + non_dyn.value
        return cost.real

    return cost_function, ansatz_state
```

### Running the optimization loop


```python
from quri_parts.algo.optimizer import NFT, OptimizerState, Optimizer, OptimizerStatus

N_DISCRETIZE = 2
N_LOCAL_QUBIT = 4
MASS = 1.0
LAM = 1.0
system = DiscreteScalarField1D(N_DISCRETIZE, N_LOCAL_QUBIT, MASS, LAM,)

n_layers = 2

def vqe(cost: CostFunction, optimizer: Optimizer, init_param: Params) -> OptimizerState:
    it = 0
    opt_state = optimizer.get_init_state(init_param)
    while opt_state.status != OptimizerStatus.CONVERGED:
        opt_state = optimizer.step(opt_state, cost)
        it += 1
        print(f"{it}-th iteration", opt_state.cost)
    return opt_state

cost_fn, ansatz_state = get_cost_function(system, n_layers)
init_param = np.random.random(ansatz_state.parametric_circuit.parameter_count)
opt_result = vqe(cost_fn, NFT(), init_param)
```
```output
1-th iteration 1.5149630657035198
2-th iteration 1.3570414047221697
3-th iteration 1.3458265324560763
4-th iteration 1.3437246180701101
5-th iteration 1.342633788950783
``````output
6-th iteration 1.3419101280151693
7-th iteration 1.3413990541889356
8-th iteration 1.3410308194876916
9-th iteration 1.3407609539962266
10-th iteration 1.340558734474652
``````output
11-th iteration 1.3404028617863892
12-th iteration 1.3402787172942934
13-th iteration 1.340176368495582
14-th iteration 1.340089119811582
15-th iteration 1.3400124794510904
``````output
16-th iteration 1.3399434356609596
17-th iteration 1.3398799576828258
18-th iteration 1.339820657313447
19-th iteration 1.3397645636718076
20-th iteration 1.3397109764585005
``````output
21-th iteration 1.339659372466516
22-th iteration 1.339609347232617
23-th iteration 1.3395605791264005
24-th iteration 1.3395128072499054
25-th iteration 1.3394658175206422
``````output
26-th iteration 1.3394194334377703
27-th iteration 1.3393735094677344
28-th iteration 1.3393279259050215
29-th iteration 1.3392825846163767
30-th iteration 1.3392374053852092
``````output
31-th iteration 1.3391923227305125
32-th iteration 1.3391472831450082
33-th iteration 1.339102242721705
34-th iteration 1.3390571651418766
35-th iteration 1.3390120199938773
``````output
36-th iteration 1.3389667813877368
37-th iteration 1.338921426828081
38-th iteration 1.3388759363076268
39-th iteration 1.3388302915853103
``````output
40-th iteration 1.3387844756165326
41-th iteration 1.338738472106688
42-th iteration 1.3386922651633477
43-th iteration 1.3386458390268172
44-th iteration 1.3385991778618034
``````output
45-th iteration 1.338552265596917
46-th iteration 1.3385050858009404
47-th iteration 1.3384576215873876
48-th iteration 1.3384098555406225
49-th iteration 1.3383617696583776
``````output
50-th iteration 1.3383133453067395
51-th iteration 1.3382645631844978
52-th iteration 1.338215403294676
53-th iteration 1.3381658449213927
54-th iteration 1.3381158666108892
``````output
55-th iteration 1.3380654461556993
56-th iteration 1.3380145605812461
57-th iteration 1.337963186134456
58-th iteration 1.3379112982738393
59-th iteration 1.3378588716610167
``````output
60-th iteration 1.3378058801532782
61-th iteration 1.3377522967971736
62-th iteration 1.3376980938230947
63-th iteration 1.337643242640637
64-th iteration 1.337587713834975
``````output
65-th iteration 1.3375314771640225
66-th iteration 1.337474501556512
67-th iteration 1.3374167551110518
68-th iteration 1.3373582050961126
69-th iteration 1.3372988179510752
``````output
70-th iteration 1.3372385592883587
71-th iteration 1.3371773938967284
72-th iteration 1.3371152857458248
73-th iteration 1.3370521979920473
74-th iteration 1.3369880929858018
``````output
75-th iteration 1.336922932280277
76-th iteration 1.336856676641841
77-th iteration 1.3367892860620756
78-th iteration 1.336720719771726
``````output
79-th iteration 1.3366509362564416
80-th iteration 1.3365798932746686
81-th iteration 1.3365075478775819
82-th iteration 1.336433856431341
``````output
83-th iteration 1.3363587746416807
84-th iteration 1.3362822575809656
85-th iteration 1.3362042597178618
86-th iteration 1.336124734949583
87-th iteration 1.3360436366370398
``````output
88-th iteration 1.3359609176426765
89-th iteration 1.335876530371306
90-th iteration 1.3357904268138003
91-th iteration 1.33570255859383
92-th iteration 1.3356128770175046
``````output
93-th iteration 1.3355213331259717
94-th iteration 1.3354278777509114
95-th iteration 1.335332461572819
96-th iteration 1.3352350351819138
97-th iteration 1.3351355491415884
``````output
98-th iteration 1.3350339540540914
99-th iteration 1.3349302006281727
100-th iteration 1.3348242397484695
101-th iteration 1.3347160225460728
``````output
102-th iteration 1.3346055004699269
103-th iteration 1.3344926253585208
104-th iteration 1.3343773495112299
105-th iteration 1.33425962575865
106-th iteration 1.3341394075311603
``````output
107-th iteration 1.3340166489248806
108-th iteration 1.3338913047640464
109-th iteration 1.333763330658852
110-th iteration 1.3336326830576168
111-th iteration 1.333499319292101
``````output
112-th iteration 1.3333631976147187
113-th iteration 1.3332242772262548
114-th iteration 1.3330825182927803
115-th iteration 1.3329378819501045
116-th iteration 1.3327903302943958
``````output
117-th iteration 1.3326398263572965
118-th iteration 1.3324863340638673
119-th iteration 1.332329818171784
120-th iteration 1.3321702441900078
121-th iteration 1.3320075782753487
``````output
122-th iteration 1.3318417871052106
123-th iteration 1.3316728377247806
124-th iteration 1.3315006973672536
125-th iteration 1.3313253332453703
``````output
126-th iteration 1.331146712312853
127-th iteration 1.3309648009943023
128-th iteration 1.3307795648823002
129-th iteration 1.330590968400419
130-th iteration 1.3303989744309828
``````output
131-th iteration 1.3302035439067452
132-th iteration 1.330004635365307
133-th iteration 1.3298022044656128
134-th iteration 1.3295962034657125
135-th iteration 1.3293865806611718
``````output
136-th iteration 1.3291732797833875
137-th iteration 1.3289562393572916
138-th iteration 1.3287353920178113
139-th iteration 1.3285106637843682
140-th iteration 1.328281973292754
``````output
141-th iteration 1.3280492309834362
142-th iteration 1.3278123382452356
143-th iteration 1.3275711865131927
144-th iteration 1.327325656318926
145-th iteration 1.327075616291554
``````output
146-th iteration 1.3268209221070448
147-th iteration 1.3265614153827878
148-th iteration 1.326296922514427
149-th iteration 1.3260272534506932
150-th iteration 1.3257522004015683
``````output
151-th iteration 1.3254715364744705
152-th iteration 1.3251850142318717
153-th iteration 1.3248923641633288
154-th iteration 1.3245932930633555
155-th iteration 1.3242874823058488
``````output
156-th iteration 1.3239745860042345
157-th iteration 1.323654229045023
158-th iteration 1.3233260049812354
159-th iteration 1.3229894737699155
160-th iteration 1.3226441593362523
``````output
161-th iteration 1.322289546944125
162-th iteration 1.3219250803505331
163-th iteration 1.3215501587175986
164-th iteration 1.32116413325237
165-th iteration 1.3207663035395742
``````output
166-th iteration 1.320355913526862
167-th iteration 1.3199321471148366
168-th iteration 1.3194941232955124
169-th iteration 1.3190408907715216
170-th iteration 1.3185714219750326
``````output
171-th iteration 1.3180846063876623
172-th iteration 1.317579243041359
173-th iteration 1.317054032052469
174-th iteration 1.3165075650070381
175-th iteration 1.3159383139718235
``````output
176-th iteration 1.3153446188504092
177-th iteration 1.3147246727347732
178-th iteration 1.3140765048146181
179-th iteration 1.3133979602958943
180-th iteration 1.312686676638861
``````output
181-th iteration 1.3119400552467695
182-th iteration 1.311155227507016
183-th iteration 1.310329013792937
184-th iteration 1.3094578736546225
185-th iteration 1.3085378449346734
``````output
186-th iteration 1.307564468898642
187-th iteration 1.3065326976162444
188-th iteration 1.3054367786890309
189-th iteration 1.3042701108797912
190-th iteration 1.303025062093686
``````output
191-th iteration 1.3016927382462347
192-th iteration 1.300262687463469
193-th iteration 1.2987225182405606
194-th iteration 1.2970574017813297
195-th iteration 1.2952494164159223
``````output
196-th iteration 1.2932766736167978
197-th iteration 1.2911121372477146
198-th iteration 1.2887220045621608
199-th iteration 1.2860634494268512
``````output
200-th iteration 1.2830814184825279
201-th iteration 1.2797039895933684
202-th iteration 1.2758354943779133
203-th iteration 1.2713460701531516
204-th iteration 1.266055342204992
``````output
205-th iteration 1.2597061475319984
206-th iteration 1.2519207814924815
207-th iteration 1.242125482134058
208-th iteration 1.2294152751494951
209-th iteration 1.2123044587869916
``````output
210-th iteration 1.188261967045551
211-th iteration 1.1529070639690504
212-th iteration 1.099164827055886
213-th iteration 1.0198728138662316
214-th iteration 0.9249611929714315
``````output
215-th iteration 0.8513364053064836
216-th iteration 0.8098988485955468
217-th iteration 0.7872861955237866
218-th iteration 0.7734341916584556
``````output
219-th iteration 0.763498334882033
220-th iteration 0.7553334517862683
221-th iteration 0.7478506521140013
222-th iteration 0.7404527170281239
223-th iteration 0.7328019117295511
``````output
224-th iteration 0.7246880523107297
225-th iteration 0.7159533042931229
226-th iteration 0.7064532172392595
227-th iteration 0.6960397141694601
228-th iteration 0.6845594491691641
``````output
229-th iteration 0.6718679110006174
230-th iteration 0.6578649264098768
231-th iteration 0.6425591767922264
232-th iteration 0.6261624381633542
233-th iteration 0.6091887880895226
``````output
234-th iteration 0.5924880731654874
235-th iteration 0.5771144630269762
236-th iteration 0.5640062579164964
237-th iteration 0.5536439928809014
238-th iteration 0.5459453814378075
``````output
239-th iteration 0.5404536862359994
240-th iteration 0.5366159039889848
241-th iteration 0.5339559544064307
242-th iteration 0.5321198586422524
``````output
243-th iteration 0.5308581344227497
244-th iteration 0.529996350514194
245-th iteration 0.5294118481543169
246-th iteration 0.5290181383139554
247-th iteration 0.5287545171051786
``````output
248-th iteration 0.5285788038585892
249-th iteration 0.5284620417967512
250-th iteration 0.5283845820227617
251-th iteration 0.5283332208676152
252-th iteration 0.5282991508574477
``````output
253-th iteration 0.5282765265257967
254-th iteration 0.5282614799961807
255-th iteration 0.5282514555477895
256-th iteration 0.5282447647274955
257-th iteration 0.528240291031197
```
We can compare the VQE state against the exact ground state of the local Hamiltonian


```python
from quri_algo.core.estimator.hadamard_test import shift_state_circuit
from quri_parts.qulacs.simulator import evaluate_state_to_vector
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh

local_state = ansatz_state.bind_parameters(opt_result.params)

local_state_vector = evaluate_state_to_vector(local_state).vector

local_hamiltonian_matrix = get_local_hamiltonian_matrix(system)
local_field_operator = get_scalar_field_operator(0, system.n_phi_qubit, system.mb)
local_field_operator_matrix = get_sparse_matrix(local_field_operator, system.n_phi_qubit)
sparse_local = csc_matrix(local_hamiltonian_matrix)
sparse_local.eliminate_zeros()

_, exact_local_gs_vec = eigsh(sparse_local, k=1, which="SA")
print(
    "Overlap with exact local Hamiltonian GS vector:",
    np.abs(exact_local_gs_vec.flatten().conj() @ local_state_vector)**2
)
print("------------------------------------------------------------")
print(
    "Exact GS energy:",
    (exact_local_gs_vec.conj().T @ local_hamiltonian_matrix @ exact_local_gs_vec)[0, 0].real
)
print("VQE energy:", cost_fn(opt_result.params))
print("------------------------------------------------------------")
print("Exact state <Φ0>:", np.round(exact_local_gs_vec.conj().T @ local_field_operator_matrix @ exact_local_gs_vec, 12)[0, 0].real)
print("VQE state <Φ0>:", estimator(local_field_operator, local_state).value.real)
```
```output
Overlap with exact local Hamiltonian GS vector: 0.9999349943025017
------------------------------------------------------------
Exact GS energy: 0.5277361259588181
VQE energy: 0.5282402910312016
------------------------------------------------------------
``````output
Exact state <Φ0>: -0.0
VQE state <Φ0>: 7.46750566180407e-05
```
Now, we build a state for the full system, i.e.

$$|\Psi\rangle = \bigotimes_{j=0}^{N_{\text{site} -1}} |\psi_j(\vec{\theta}^*)\rangle$$


```python
init_state_circuit = QuantumCircuit(system.n_state_qubit)
for site in range(system.n_discretize):
    shift = site * system.n_phi_qubit
    local_circuit = shift_state_circuit(local_state.circuit, shift)
    init_state_circuit.extend(local_circuit)

INIT_STATE = quantum_state(system.n_state_qubit, circuit=init_state_circuit)
```

We may examine some properties of the prepared state


```python
from quri_parts.qulacs.simulator import evaluate_state_to_vector
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
import itertools
import functools

init_state_vector = evaluate_state_to_vector(INIT_STATE).vector

full_hamiltonian = get_hamiltonian_matrix(system)
sparse_full = csc_matrix(full_hamiltonian)
sparse_full.eliminate_zeros()

exact_gs_energy, exact_gs_vec = eigsh(sparse_full, k=1, which="SA")


field_operator = functools.reduce(
    lambda a, b: a + b,
    [get_scalar_field_operator(n, system.n_phi_qubit, system.mb) for n in range(system.n_discretize)]
)
field_operator_matrix = get_sparse_matrix(field_operator).toarray()

print("Overlap with exact GS:", np.abs(exact_gs_vec.flatten().conj() @ init_state_vector)**2)
print("----------------------------------------------------------------")
print("Exact GS energy:", exact_gs_energy[0])
print("Prepared state <H>:", (init_state_vector.conj() @ full_hamiltonian @ init_state_vector).real)
print("----------------------------------------------------------------")
print("Exact <Φ>:", (exact_gs_vec.flatten().conj() @ field_operator_matrix @ exact_gs_vec).real[0])
print("Prepared <Φ>:", (init_state_vector.conj() @ field_operator_matrix @ init_state_vector).real)
```
```output
Overlap with exact GS: 0.9720739298079517
----------------------------------------------------------------
Exact GS energy: 1.4021160736167078
Prepared state <H>: 1.5067423412222323
----------------------------------------------------------------
Exact <Φ>: 1.4051260155412137e-16
Prepared <Φ>: 0.00014935011323622543
```
There's a nonzero $\langle \Phi \rangle$ for the prepared state, so there should be non-trivial time evolution. The ground state with a non-trivial VEV needs to be prepared with adiabatic time evolution, which is not covered in this notebook, but is introduced in section IV of [1].

## Time Evolution

Here, we evolve the prepared state from the last section.

### Trotter time evolution

It is generally hard to build the exact time evolution due to the non-trivial commutator $[\Phi_i, \Pi_j]$. Usually, Trotterization is used to approximate the exact time evolution on a quantum computer:

$$
e^{-iHt} \approx \prod_{i=1}^{N}\prod_{j=0}^{N_{\text{site}}-1} e^{-\frac{i}{2}\Pi_j^2\frac{t}{N}} e^{-\frac{im^2}{2}\Phi_j^2\frac{t}{N}} e^{-\frac{i\lambda}{4!}\Phi_j^4\frac{t}{N}} e^{-iJ\Phi_j\frac{t}{N}} e^{-\frac{i}{2}(\Phi_j - \Phi_{j+1})^2\frac{t}{N}}
$$

In QURI Algo, circuit implementation of time evolution operators are represented by subclasses of `TimeEvolutionCircuitFactory`, where we build one for Trotterization implementation of the time evolution operator.


```python
import math
from quri_algo.circuit.time_evolution.interface import TimeEvolutionCircuitFactory
from quri_algo.circuit.utils.transpile import apply_transpiler
from quri_parts.circuit import NonParametricQuantumCircuit, LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.circuit import add_parametric_commuting_paulis_exp_gate

@dataclass
class DiscreteScalarFieldTrotterTimeEvoFactory(TimeEvolutionCircuitFactory):

    system: DiscreteScalarField1D
    n_trotter: int

    def __post_init__(self) -> None:
        self._evo_circuit = LinearMappedUnboundParametricQuantumCircuit(
            self.system.n_state_qubit
        )
        self._t = self._evo_circuit.add_parameter("t")
        self._construct_evolution_circuit()

    def field_operator(self, site_label: int) -> Operator:
        """The field operator on a given site.
        """
        return get_scalar_field_operator(
            site_label, self.system.n_phi_qubit, self.system.mb
        )

    def _construct_evolution_circuit(self) -> None:
        for _ in range(self.n_trotter):
            self._add_conj_momentum_square_evolution()
            self._add_ext_field_evolution()
            self._add_kinetic_evolution()
            self._add_phi_square_evolution()
            self._add_phi4_evolution()

    def _add_ext_field_evolution(self) -> NonParametricQuantumCircuit:
        """Add On site Φ_j^2 evolution term
        """
        J = self.system.external_field
        n_site = self.system.n_discretize
        for site in range(n_site):
            field_operator = self.field_operator(site)
            operator = J * field_operator
            add_parametric_commuting_paulis_exp_gate(
                self._evo_circuit, {self._t: -1.0/self.n_trotter}, operator,
            )

    def _add_phi_square_evolution(self) -> NonParametricQuantumCircuit:
        """Add On site Φ_j^2 evolution term
        """
        mb = self.system.mb
        n_site = self.system.n_discretize
        for site in range(n_site):
            field_operator = self.field_operator(site)
            operator = 0.5 * mb**2 * field_operator * field_operator
            add_parametric_commuting_paulis_exp_gate(
                self._evo_circuit, {self._t: -1.0/self.n_trotter}, operator,
            )

    def _add_conj_momentum_square_evolution(self) -> None:
        """On site Π_j^2 evolution term.
        """
        mb = self.system.mb
        f_gate = QuantumCircuit(
            self.system.n_state_qubit, gates=create_F_gate(n_qubits).gates
        )
        n_site = self.system.n_discretize
        n_site_qubit = self.system.n_phi_qubit
        for site in range(n_site):
            site_f = shift_state_circuit(f_gate, shift=site * n_site_qubit)
            inverse_f = inverse_circuit(site_f)
            field_operator = self.field_operator(site)
            operator = 0.5 * mb**2 * field_operator * field_operator
            self._evo_circuit.extend(inverse_f.gates)
            add_parametric_commuting_paulis_exp_gate(
                self._evo_circuit, {self._t: -1.0/self.n_trotter}, operator,
            )
            self._evo_circuit.extend(site_f.gates)

    def _add_phi4_evolution(self) -> None:
        """Add Φ_j^4 evolution term to circuit.
        """
        lam = self.system.lam
        n_site = self.system.n_discretize
        for site in range(n_site):
            field_operator = self.field_operator(site)
            operator = (lam / math.factorial(4)) * field_operator * field_operator * field_operator * field_operator
            add_parametric_commuting_paulis_exp_gate(
                self._evo_circuit, {self._t: -1.0/self.n_trotter}, operator,
            )

    def _add_kinetic_evolution(self) -> None:
        """Add (Φ_j - Φ_{j+1})^2 evolution term to circuit.
        """
        n_site = self.system.n_discretize
        for site in range(n_site - 1):
            field_operator = self.field_operator(site)
            next_field_operator = self.field_operator(site + 1)
            diff = next_field_operator - field_operator
            operator = 0.5 * diff * diff
            add_parametric_commuting_paulis_exp_gate(
                self._evo_circuit, {self._t: -1.0/self.n_trotter}, operator,
            )


    # @apply_transpiler
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        return self._evo_circuit.bind_parameters([evolution_time])

```

### Exact time evolution

We can also build one exact time evolution operator using one unitary matrix gate


```python
import functools
import numpy.typing as npt
from scipy.linalg import expm


@dataclass
class DiscreteScalarFieldExactTimeEvoFactory(TimeEvolutionCircuitFactory):

    system: DiscreteScalarField1D

    @functools.cached_property
    def hamiltonian_matrix(self) -> npt.NDArray[np.complex128]:
        return get_hamiltonian_matrix(self.system)

    def get_time_evolution_matrix(self, evolution_time: float) -> npt.NDArray[np.complex128]:
        return expm(-1j * evolution_time * self.hamiltonian_matrix)

    # @apply_transpiler
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        n_qubits = self.system.n_state_qubit
        evo = self.get_time_evolution_matrix(evolution_time)
        circuit = QuantumCircuit(self.system.n_state_qubit)
        circuit.add_UnitaryMatrix_gate(list(range(n_qubits)), evo)
        return circuit

```

### Execute the time evolution circuits

With all the components defined in the previous sections, we can build the time evolution circuit factories for both Trotter time evolution and exact time evolution.


```python
n_trotter = 10
trotter_evo_factory = DiscreteScalarFieldTrotterTimeEvoFactory(system, n_trotter=n_trotter)
exact_evo_factory = DiscreteScalarFieldExactTimeEvoFactory(system)

```

Here we look at the time dependence of the field operator $\Phi$. As explained in the state preparation section, it has a small but non-zero initial $\langle\Phi\rangle$, so we would expect $\Phi$ fluctuate with time with a small amplitude.


```python
import functools
field_operator = functools.reduce(
    lambda a, b: a + b,
    [get_scalar_field_operator(n, system.n_phi_qubit, system.mb) for n in range(system.n_discretize)]
)
```


```python
import tqdm

ts = np.linspace(0, 10, 101)

trotter_phi_exp_val = []
exact_phi_exp_val = []

for t in tqdm.tqdm(ts):
    trotter_evo = trotter_evo_factory(t)
    trotter_state = INIT_STATE.with_gates_applied(trotter_evo)
    trotter_phi_exp_val.append(estimator(field_operator, trotter_state).value.real)

    exact_evo = exact_evo_factory(t)
    exact_state = INIT_STATE.with_gates_applied(exact_evo)
    exact_phi_exp_val.append(estimator(field_operator, exact_state).value.real)
```
```output
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 101/101 [00:24<00:00,  4.11it/s]
```

```python
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 8)
plt.plot(ts, np.array(exact_phi_exp_val), "-o", label="Exact time evolution")
plt.plot(ts, np.array(trotter_phi_exp_val), "-o", label="Trotter")
plt.xlabel("evolution time", size=18)
plt.ylabel("$\\langle \\Phi \\rangle$", size=18)
plt.tick_params("both", labelsize=12)
plt.title("Evolution of scalar field", size=16)
plt.legend(fontsize=12, loc="lower right")
plt.show()

```


    
![png](./index_47_0.png)
    


Here, we can see the that the exact time evolution curve suggests that the field operator oscillates around 0 with a small amplitude. The Trotter time evolution closely mimics the exact time evolution when the evolution time is small. After $t \geq 7$, the Trotter line behaves irregularly because the Trotter error begins to dominate. This can be fixed by using higher Trotter steps but comes at a price of deeper circuit, where noise from the quantum computer dominates over.

## Resource estimation

In the previous sections, we used a system of merely 2 sites and 4 qubits per site. While this is laptop-simulatable, it is very far from the realistic case where one would hope to have as many sites as possible and the subspace persite is as many as possible. Although this is not simulatable, we can still do resource estimation on how many resources a large scale simulation would take and how accurate the simulation can be on a given device or error correction architecture. Here, we use QURI VM for this purpose. Now we build a VM for square lattice superconducting NISQ device and another VM for STAR.

### NISQ device

For the square lattice NISQ device, we assume there is a $12 \times 12 = 144$ qubits device with basis gates being (`RZ`, `SqrtX`, `X`, `CNOT`). We further assume the 2-qubit gate error rate being $3 \times 10^{-3}$ and single qubit error rate being $5 \times 10^{-4}$.


```python
from quri_parts.backend.devices import nisq_spcond_lattice
from quri_parts.backend.units import TimeValue, TimeUnit
from quri_parts.circuit.topology.square_lattice import SquareLattice
from quri_vm.vm import VM

square_lattice = SquareLattice(12, 12)

nisq_property = nisq_spcond_lattice.generate_device_property(
    lattice=square_lattice,
    native_gates_1q=("RZ", "SqrtX", "X"),
    native_gates_2q=("CNOT",),
    gate_error_1q=5e-4,
    gate_error_2q=3e-3,
    gate_error_meas=2e-2,
    gate_time_1q=TimeValue(60, TimeUnit.NANOSECOND),
    gate_time_2q=TimeValue(660, TimeUnit.NANOSECOND),
    gate_time_meas=TimeValue(1.4, TimeUnit.MICROSECOND)
)

nisq_vm = VM.from_device_prop(nisq_property)
```

### Partial error correction device: STAR architecture

Now, we build a VM for an error correction device based on the STAR architecture. This is a partial error correcting architecture introduced in [3]. We assume only 50 qubits are available to us, the code distance is 9 and the physical error rate is $p_{phys} = 10^{-4}$. The basis gates are `CNOT`, `RZ`, `H` and `S`, where the Clifford gates `CNOT`, `H` and `S` are fully error corrections and the `RZ` gate is subjected to logical error rate of:

$$P = 1 - \left(1 - \frac{2p_{phys}}{15}\right)^2.$$


```python
from quri_parts.backend.devices import star_device

logical_qubit_count = 50
p_phys = 1e-4
star_property = star_device.generate_device_property(
    logical_qubit_count,
    code_distance=9,
    qec_cycle=TimeValue(1, TimeUnit.MICROSECOND),
    physical_error_rate=p_phys
)
star_vm = VM.from_device_prop(star_property)
```

We first estimate the number of physical qubits needed to implement such device.


```python
print(f"{star_property.physical_qubit_count} physical qubits used to implement a device of 50 logical qubits and code distance 9.")
```
```output
32400 physical qubits used to implement a device of 50 logical qubits and code distance 9.
```
### Resourse estimation: small scale 

Here, we evaluate the circuit execution time and circuit fidelity of the both the state preparation circuit and the time evolution circuit for the 8-qubit system used through out the previous sections.


#### State preparation

First, we start with the state preparation circuit. Recall that we used the 2-layer hardware efficient ansatz to perform the VQE computation. The 2-qubit gates in the hardware efficient ansatz only act on adjacent qubits, so is suitable to be executed on NISQ devices.


```python
from pprint import pprint

print("NISQ device analysis")
pprint(nisq_state_prep := nisq_vm.analyze(init_state_circuit))

print("----------------------------------------------------------------------------------")
print("STAR device analysis")
pprint(star_state_prep := star_vm.analyze(init_state_circuit))
```
```output
NISQ device analysis
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=8,
              gate_count=276,
              depth=51,
              latency=TimeValue(value=5460.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=0.8452801913024685)
----------------------------------------------------------------------------------
STAR device analysis
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=8,
              gate_count=204,
              depth=30,
              latency=TimeValue(value=747000.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=0.9986859607354742)
```
As we can see above, the fidelity of the state preparation circuit on the NISQ device is about 84%, so we can expect the state preparation to be quite accurate even on today's devices with suitable error mitigation. For the STAR architecture, the circuit fidelity reaches fidelity over 99%. However, it comes at a price of execution time over 100 times. While execution time of a single circuit takes merely about 0.7 microseconds, the time of running the whole VQE loop will be magnified, which we will now estimate.

Recall that each call to the cost function involves 2 circuit execution, one pure 4-qubit hardware effecient circuit and the other 4-qubit hardware effecient circuit extended with an $F^{\dagger}$ operation.


```python
def estimate_vqe_time(
    vm: VM, system: DiscreteScalarField1D, vqe_result: OptimizerState, n_shots_per_iter: int
) -> tuple[TimeValue, float, float]:
    vqe_circuit = HardwareEfficient(system.n_phi_qubit, n_layers)
    conj_momentum_vqe_circuit = vqe_circuit + create_F_gate(system.n_phi_qubit)

    phi_analysis = vm.analyze(vqe_circuit)
    pi_analysis = vm.analyze(conj_momentum_vqe_circuit)

    one_call_time = phi_analysis.latency.in_ns() + pi_analysis.latency.in_ns()

    return (
        TimeValue(one_call_time * vqe_result.funcalls * 1e-9 * n_shots_per_iter, TimeUnit.SECOND),
        phi_analysis.fidelity,
        pi_analysis.fidelity,
    )
```

Assuming each iterations take $10^5$ shots


```python
n_shots = int(1e5)
nisq_vqe_time, nisq_phi_fid, nisq_pi_fid = estimate_vqe_time(nisq_vm, system, opt_result, n_shots)
star_vqe_time, star_phi_fid, star_pi_fid = estimate_vqe_time(star_vm, system, opt_result, n_shots)
print(
    f"""
    VQE takes {nisq_vqe_time.value / 3600: .3f} hours to be executed on NISQ device.
    (|Ψ> fidelity: {nisq_phi_fid}, F†|Ψ> fidelity: {nisq_pi_fid})
    """
)
print(
    f"""
    VQE takes {star_vqe_time.value / 3600: .3f} hours to be executed on STAR device.
    (|Ψ> fidelity: {star_phi_fid}, F†|Ψ> fidelity: {star_pi_fid})
    """
)
```
```output

    VQE takes  25.789 hours to be executed on NISQ device.
    (|Ψ> fidelity: 0.9417291638110525, F†|Ψ> fidelity: 0.6746884636761388)
    

    VQE takes  388.399 hours to be executed on STAR device.
    (|Ψ> fidelity: 0.9999951651847668, F†|Ψ> fidelity: 0.9993389831048093)
```
Here, we can see that executing VQE on an error corrected device takes almost 20 times longer.

#### Time evolution

Next, we evaluate the time evolution circuit on both NISQ and STAR devices.


```python
print("NISQ device analysis")
pprint(nisq_vm.analyze(init_state_circuit + trotter_evo_factory(10)))

print("----------------------------------------------------------------------------------")
print("STAR device analysis")
pprint(star_vm.analyze(init_state_circuit + trotter_evo_factory(10)))
```
```output
NISQ device analysis
``````output
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=8,
              gate_count=18326,
              depth=10594,
              latency=TimeValue(value=6602760.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=2.2073225801174015e-21)
----------------------------------------------------------------------------------
STAR device analysis
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=8,
              gate_count=4344,
              depth=1324,
              latency=TimeValue(value=30807000.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=0.9638181279070184)
```
As shown in the results above, the fidelity of running the time evolution circuit on a NISQ device is close to 0 while the fidelity on STAR device remains over 96%. This gives error corrected device major advantage for simulating the scalar field theory.

### Resource estimation: large scale 

The previous section serves as a toy example of cost estimation. We now move to large scale systems that is not laptop-simulatable and estimate the cost like we did in the last section. We discretize the scalar field to 8 sites and each site contains 6 qubits. This gives as 48 logical qubits in total and is almost the maximal size the STAR device can hold.


```python
large_system = DiscreteScalarField1D(n_discretize=8, n_phi_qubit=6, mb=1, lam=1)
```

#### State preparation cost

As in the small scale section, we first estimate the fidelity and execution time of the initial state circuit and estimate the cost of executing VQE.


```python
large_vqe_ansatz = HardwareEfficient(large_system.n_phi_qubit, reps=2)
large_vqe_circuit = large_vqe_ansatz.bind_parameters(np.random.random(large_vqe_ansatz.parameter_count))

large_init_circuit = QuantumCircuit(large_system.n_state_qubit)
for site in range(large_system.n_discretize):
    shift = site * large_system.n_phi_qubit
    large_init_circuit.extend(shift_state_circuit(large_vqe_circuit, shift).gates)

large_init_circuit
```



```output
<quri_parts.rust.circuit.circuit.QuantumCircuit at 0x7f17080f97d0>
```



```python
pprint("NISQ device analysis")
pprint(nisq_vm.analyze(large_init_circuit))

print("----------------------------------------------------------------------------------")
pprint("STAR device analysis")
pprint(star_vm.analyze(large_init_circuit))
```
```output
'NISQ device analysis'
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=48,
              gate_count=1776,
              depth=51,
              latency=TimeValue(value=5460.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=0.3366966870560188)
----------------------------------------------------------------------------------
'STAR device analysis'
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=48,
              gate_count=1248,
              depth=30,
              latency=TimeValue(value=747000.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=0.992141619566363)
```
Next, we again estimate the cost of running VQE.


```python
n_shots = int(1e5)
large_nisq_vqe_time, large_nisq_phi_fid, large_nisq_pi_fid = estimate_vqe_time(nisq_vm, large_system, opt_result, n_shots)
large_star_vqe_time, large_star_phi_fid, large_star_pi_fid = estimate_vqe_time(star_vm, large_system, opt_result, n_shots)
print(
    f"""
    VQE takes {large_nisq_vqe_time.value / 3600: .3f} hours to be executed on NISQ device.
    (|Ψ> fidelity: {large_nisq_phi_fid}, F†|Ψ> fidelity: {large_nisq_pi_fid})
    """
)
print(
    f"""
    VQE takes {large_star_vqe_time.value / 3600: .3f} hours to be executed on STAR device.
    (|Ψ> fidelity: {large_star_phi_fid}, F†|Ψ> fidelity: {large_star_pi_fid})
    """
)
```
```output

    VQE takes  85.072 hours to be executed on NISQ device.
    (|Ψ> fidelity: 0.9047807805707612, F†|Ψ> fidelity: 0.2691948085794522)
    

    VQE takes  538.747 hours to be executed on STAR device.
    (|Ψ> fidelity: 0.999992747785916, F†|Ψ> fidelity: 0.9984874866534053)
```
### Time evolution circuit

Finally, we evaluate the fidelity of the time evolution circuit.


```python
trotter_step = 10
large_time_evolution_factory = DiscreteScalarFieldTrotterTimeEvoFactory(large_system, trotter_step)
evo_circuit = large_time_evolution_factory(evolution_time=1.0)
```


```python
print("NISQ device analysis")
pprint(nisq_vm.analyze(evo_circuit))

print("----------------------------------------------------------------------------------")
print("STAR device analysis")
pprint(star_vm.analyze(evo_circuit))
```
```output
NISQ device analysis
``````output
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=48,
              gate_count=369060,
              depth=100168,
              latency=TimeValue(value=65684280.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=8.2e-322)
----------------------------------------------------------------------------------
STAR device analysis
``````output
AnalyzeResult(lowering_level=<LoweringLevel.ArchLogicalCircuit: 1>,
              qubit_count=48,
              gate_count=41060,
              depth=3028,
              latency=TimeValue(value=69984000.0, unit=<TimeUnit.NANOSECOND>),
              fidelity=0.6885154917215952)
```
This shows that simulating a 48 qubit time evolution with Trotterization is practically impossible, and we need to resort to an error correcting device for better fidelity.

## Hilbert-Schmidt test

While the field operator expectation value for short evolution times closely follows the exact time evolution as seen above, we can confirm that the time-evolution operator as a whole is faithful to this behaviour by evaluating the Hilbert-Schmidt test

$$
    C_\textup{HST}(U,V) = 1 - \frac{1}{4^{N_q}} |\textup{Tr}[U^\dagger V]|^2.
$$

The rightmost term above, the Hilbert-Schmidt product, is equivalent to the average fidelity of two copies of a Haar random state undergoing unitary rotation by $U$ and $V$.

QURI Algo let's us do this with ease. What we need is to import the `HilberSchmidtTest` class from `quri_algo.core.cost_functions` module


```python
import tqdm
from quri_algo.core.cost_functions import HilbertSchmidtTest

hs_test = HilbertSchmidtTest(
    estimator, alpha=1.0  # This alpha interpolates between the local and global HS tests
)
lhs_test = HilbertSchmidtTest(
    estimator, alpha=0.0
)

ts = np.linspace(0, 10, 101)
hs_cost = []
lhs_cost = []

for t in tqdm.tqdm(ts):
    trotter_evo = trotter_evo_factory(t)
    trotter_state = INIT_STATE.with_gates_applied(trotter_evo)

    exact_evo = exact_evo_factory(t)
    exact_state = INIT_STATE.with_gates_applied(exact_evo)

    hs_cost.append(hs_test(exact_state.circuit, trotter_state.circuit).value.real)
    lhs_cost.append(lhs_test(exact_state.circuit, trotter_state.circuit).value.real)
```
```output
100%|██████████████████████████████████████████████████████████████████████████████| 101/101 [04:42<00:00,  2.80s/it]
```
We will plot the cost-function below for each time-step


```python

plt.rcParams["figure.figsize"] = (12, 8)
plt.plot(ts, np.array(exact_phi_exp_val), "-o", label="Exact time evolution")
plt.plot(ts, np.array(trotter_phi_exp_val), "-o", label="Trotter")
plt.xlabel("evolution time", size=18)
plt.ylabel("$\\langle \\Phi \\rangle$", size=18)
plt.tick_params("both", labelsize=12)
plt.title("Evolution of scalar field", size=16)
plt.twinx()
plt.plot(ts, np.array(hs_cost), "-or", label="Hilbert-Schmidt cost")
plt.plot(ts, np.array(lhs_cost), "-om", label="Local Hilbert-Schmidt cost")
plt.ylabel("$C_\\text{HST}(U,V)$", size=18)
plt.legend(fontsize=12, loc="upper right")
plt.show()
```


    
![png](./index_81_0.png)
    


Evidently, the fidelity of two unitaries is a much stricter condition for accurate time-evolution compared to the operator expectation value - and this makes sense since the expectation value of a quantum operator with respect to a pure state is only sensitive to a single state, whereas the Hilbert-Schmidt product is a state averaged quantity over the whole Hilbert space!

Does the situation change if we use the local Hilbert-Schmidt test? Well, to some extent the local Hilbert-Schmidt cost is lower. This is because it is an average over local cost-functions that each only depend on the resulting density matrix of a single pair of qubits. Here each of these local cost functions measure whether those qubit pairs can be disentangled or by a local operation. This defines the entanglement fidelity, which is weighted equally for each pair. In contrast, for the global Hilbert-Schmidt cost to be significantly lower than unity every qubit pair must simultaneously be disentangled.

The main take-away is that the two cost-functions, the local and the global Hilbert-Schmidt tests in general do not have the same inflection point, and in general for larger systems the local Hilbert Schmidt test will be smoother, whereas the Hilbert-Schmidt product for the global one will be exponentially suppressed. Thus for larger qubit systems the local one is preferred as it is more likely to have non-zero gradients when used as a cost-function

### Can we improve it by adding more Trotter steps?

Let's try with more Trotter steps and see if the situation improves. We will add several Trotter steps below and evaluate the Hilbert-Schmidt cost of each circuit. For the evaluation we will compare with the exact time-evolution unitary.


```python

trotter_steps = [20, 30, 40, 50, 60]
exact_evo_factory = DiscreteScalarFieldExactTimeEvoFactory(system)

hs_cost_lists= []

for i, n_trotter in enumerate(tqdm.tqdm(trotter_steps)):
    trotter_evo_factory = DiscreteScalarFieldTrotterTimeEvoFactory(system, n_trotter)
    hs_cost_lists.append([])
    for t in tqdm.tqdm(ts):
        trotter_evo = trotter_evo_factory(t)
        trotter_state = INIT_STATE.with_gates_applied(trotter_evo)

        exact_evo = exact_evo_factory(t)
        exact_state = INIT_STATE.with_gates_applied(exact_evo)

        hs_cost_lists[i].append(hs_test(exact_state.circuit, trotter_state.circuit).value.real)
```
```output
100%|█████████████████████████████████████████████████████████████████████████████████| 5/5 [24:28<00:00, 293.62s/it]
``````output
100%|██████████████████████████████████████████████████████████████████████████████| 101/101 [05:00<00:00,  2.98s/it]
```

```python
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (12, 8)
for hs_cost, n_trotter in zip(hs_cost_lists, trotter_steps):
    plt.plot(ts, np.array(hs_cost), "-o", label="$N_\\text{Trotter} = $"+ str(n_trotter))

plt.title("Evolution of scalar field", size=16)
plt.xlabel("evolution time", size=18)
plt.ylabel("$C_\\text{HST}(U,V)$", size=18)
plt.tick_params("both", labelsize=12)
plt.legend(fontsize=12, loc="upper right")
plt.show()
```


    
![png](./index_85_0.png)
    


It seems like this reduces the Hilbert-Schmidt cost. The inflection point for the cost-function curve moves beyond an evolution time of 5.0 when the number of Trotter steps is 60.

With reference to the devices we defined earlier, however, if we look at the circuit depth we find that


```python
print(f"The Trotterized circuit with 60 Trotter steps has depth {star_vm.analyze(trotter_state.circuit).depth} on the STAR device")
print(f"The Trotterized circuit with 60 Trotter steps has depth {nisq_vm.analyze(trotter_state.circuit).depth} on the NISQ device")
```
```output
The Trotterized circuit with 60 Trotter steps has depth 7482 on the STAR device
``````output
The Trotterized circuit with 60 Trotter steps has depth 63145 on the NISQ device
```
### Noisy simulation using QURI VM

This will obviously be difficult to run with any reasonable fidelity. So how do we know if this could even work? Let's try to use a real device again with some finite physical error rate.


```python
n_trotter = 60
t = 5.0
p_phys_range = np.logspace(-6,-4,30)
trotter_evo_factory = DiscreteScalarFieldTrotterTimeEvoFactory(system, n_trotter)
trotter_circuit = trotter_evo_factory(t)
```


```python
fidelity = []
for p_phys in tqdm.tqdm(p_phys_range):
    logical_qubit_count = system.n_state_qubit
    star_property = star_device.generate_device_property(
        logical_qubit_count,
        code_distance=9,
        qec_cycle=TimeValue(1, TimeUnit.MICROSECOND),
        physical_error_rate=p_phys
    )
    star_vm = VM.from_device_prop(star_property)

    fidelity.append(star_vm.analyze(trotter_circuit).fidelity)
```
```output
100%|████████████████████████████████████████████████████████████████████████████████| 30/30 [00:58<00:00,  1.96s/it]
```

```python
plt.rcParams["figure.figsize"] = (12, 8)
plt.semilogx(p_phys_range, np.array(fidelity), "-o", label="Quantum state fidelity")

plt.title("60 Trotter step time-evolution", size=16)
plt.xlabel("physical error rate", size=18)
plt.ylabel("Circuit fidelity", size=18)
plt.tick_params("both", labelsize=12)
plt.legend(fontsize=12, loc="upper left")
plt.show()
```


    
![png](./index_91_0.png)
    


With the partial error correction of the STAR architecture it appears that even for relatively high physical error rates we can still expect the quantum circuit to be executable in spite of its depth. Thus we have found a set of parameters where we can execute accurate time evolution using a quantum circuit representation of time-evolution for the scalar-field theory!

## Reference

[1] Andy C. Y. Li, Alexandru Macridin, Stephen Mrenna, Panagiotis Spentzouris. [Simulating scalar field theories on quantum computers with limited resources](https://arxiv.org/pdf/2210.07985)

[2] Alexandru Macridin, Andy C. Y. Li, Stephen Mrenna, Panagiotis Spentzouris. [Bosonic field digitization for quantum computers](https://arxiv.org/pdf/2108.10793)

[3] Yutaro Akahoshi, Kazunori Maruyama, Hirotaka Oshima, Shintaro Sato, Keisuke Fujii. [Partially Fault-tolerant Quantum Computing Architecture with Error-corrected Clifford Gates and Space-time Efficient Analog Rotations](https://arxiv.org/abs/2303.13181)
