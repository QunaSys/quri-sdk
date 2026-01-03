# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from quri_parts.circuit import (
    CNOT,
    RX,
    ImmutableBoundParametricQuantumCircuit,
    ImmutableParametricQuantumCircuit,
    ParametricPauliRotation,
    ParametricQuantumCircuit,
    ParametricRX,
    QuantumCircuit,
    QuantumGate,
    X,
)

_GATES: list[QuantumGate] = [
    X(0),
    RX(0, angle=1.0),
    CNOT(0, 1),
]

_PARAMS: list[float] = [1.0, 0.5]


def mutable_circuit() -> ParametricQuantumCircuit:
    circuit = ParametricQuantumCircuit(2)
    for gate in _GATES:
        circuit.add_gate(gate)
    circuit.add_ParametricRX_gate(0)
    circuit.add_ParametricPauliRotation_gate([1], [1])
    return circuit


def immutable_circuit() -> ImmutableParametricQuantumCircuit:
    q_circuit = mutable_circuit()
    return ImmutableParametricQuantumCircuit(q_circuit)


class TestUnboundParametricQuantumCircuit:
    def test_unbound_parametric_quantum_circuit(self) -> None:
        circuit = mutable_circuit()
        assert circuit.qubit_count == 2
        assert len(circuit._gates) == 5
        assert len(circuit._params) == 2
        assert circuit.has_trivial_parameter_mapping

    def test_get_mutable_copy(self) -> None:
        circuit = mutable_circuit()
        circuit_copied = circuit.get_mutable_copy()
        assert isinstance(circuit_copied, ParametricQuantumCircuit)
        assert id(circuit) != id(circuit_copied)
        assert circuit == circuit_copied

    def test_freeze(self) -> None:
        circuit = mutable_circuit()
        immut_circuit = circuit.freeze()
        assert isinstance(immut_circuit, ImmutableParametricQuantumCircuit)
        assert circuit == immut_circuit

    def test_bind_parameters(self) -> None:
        circuit = mutable_circuit()
        circuit_bound = circuit.bind_parameters(_PARAMS)
        assert isinstance(circuit_bound, ImmutableBoundParametricQuantumCircuit)
        assert all([isinstance(gate, QuantumGate) for gate in circuit_bound.gates])
        assert circuit_bound.gates[3].params[0] == 1.0
        assert circuit_bound.gates[4].params[0] == 0.5

    def test_bind_parameters_by_dict(self) -> None:
        circuit = ParametricQuantumCircuit(2)
        for gate in _GATES:
            circuit.add_gate(gate)
        theta = circuit.add_ParametricRX_gate(0)
        phi = circuit.add_ParametricPauliRotation_gate([1], [1])
        param_vals = np.random.random(2).tolist()
        param_dict = {theta: param_vals[0], phi: param_vals[1]}

        expected_circuit = circuit.bind_parameters(param_vals)
        circuit_bound = circuit.bind_parameters_by_dict(param_dict)
        assert expected_circuit.gates == circuit_bound.gates

    def test_depth(self) -> None:
        circuit = mutable_circuit()
        assert circuit.depth == 4
        immut_circuit = circuit.freeze()
        assert immut_circuit.depth == 4
        circuit_bound = circuit.bind_parameters(_PARAMS)
        assert circuit_bound.depth == 4

        circuit_2 = ParametricQuantumCircuit(3)
        circuit_2.add_ParametricRX_gate(0)
        circuit_2.add_CNOT_gate(0, 2)
        circuit_2.add_ParametricRX_gate(1)
        assert circuit_2.depth == 2
        immutable_circuit_2 = circuit_2.freeze()
        assert immutable_circuit_2.depth == 2

    def test_add(self) -> None:
        circuit = mutable_circuit()
        qc_circuit = QuantumCircuit(2)
        qc_circuit.add_H_gate(0)
        got_circuit = circuit + qc_circuit
        exp_circuit = circuit.get_mutable_copy()
        exp_circuit.add_H_gate(0)
        assert got_circuit.gates == exp_circuit.gates

        circuit = mutable_circuit()
        up_circuit = ParametricQuantumCircuit(2)
        up_circuit.add_ParametricRX_gate(0)
        got_circuit = circuit + up_circuit
        exp_circuit = circuit.get_mutable_copy()
        exp_circuit.add_ParametricRX_gate(0)
        assert got_circuit.gates == exp_circuit.gates

    def test_radd(self) -> None:
        circuit = mutable_circuit()
        qc_circuit = QuantumCircuit(2)
        qc_circuit.add_H_gate(0)
        got_circuit = qc_circuit + circuit
        exp_circuit = ParametricQuantumCircuit(2)
        exp_circuit.add_H_gate(0)
        for gate in _GATES:
            exp_circuit.add_gate(gate)
        exp_circuit.add_ParametricRX_gate(0)
        exp_circuit.add_ParametricPauliRotation_gate([1], [1])
        assert got_circuit.gates == exp_circuit.gates

    def test_add_gate(self) -> None:
        circuit = mutable_circuit()
        assert circuit.gates == _GATES + [
            ParametricRX(0),
            ParametricPauliRotation([1], [1]),
        ]
        circuit.add_gate(X(0), 0)
        assert circuit.gates == [X(0)] + _GATES + [
            ParametricRX(0),
            ParametricPauliRotation([1], [1]),
        ]

    def test_sample(self) -> None:
        circuit = ParametricQuantumCircuit(3)
        circuit.add_ParametricRX_gate(0)
        circuit.add_CNOT_gate(0, 2)
        samples = circuit.sample(1000, [np.pi / 4])
        assert len(samples) == 2
        assert sum(samples.values()) == 1000


class TestImmutableUnboundParametricQuantumCircuit:
    def test_immutable_unbound_parametric_quantum_circuit(self) -> None:
        circuit = immutable_circuit()
        assert circuit.qubit_count == 2
        assert len(circuit._gates) == 5
        assert len(circuit._params) == 2
        assert circuit.has_trivial_parameter_mapping

    def test_get_mutable_copy(self) -> None:
        circuit = immutable_circuit()
        immut_circuit = circuit.get_mutable_copy()
        assert isinstance(immut_circuit, ParametricQuantumCircuit)
        assert id(circuit) != id(immut_circuit)
        assert circuit == immut_circuit

    def test_freeze(self) -> None:
        circuit = immutable_circuit()
        immut_circuit = circuit.freeze()
        assert isinstance(immut_circuit, ImmutableParametricQuantumCircuit)
        assert circuit == immut_circuit

    def test_bind_parameters(self) -> None:
        circuit = immutable_circuit()
        circuit_bound = circuit.bind_parameters(_PARAMS)
        assert isinstance(circuit_bound, ImmutableBoundParametricQuantumCircuit)
        assert all([isinstance(gate, QuantumGate) for gate in circuit_bound.gates])
        assert circuit_bound.gates[3].params[0] == 1.0
        assert circuit_bound.gates[4].params[0] == 0.5


class TestParametricQuantumCircuitHashing:
    def test_parametric_quantum_circuit_hashable(self) -> None:
        """Test that ParametricQuantumCircuit can be used in sets and as dict
        keys."""
        circuit1 = mutable_circuit()
        circuit2 = mutable_circuit()

        # Test that equal circuits have the same hash
        assert hash(circuit1.freeze()) == hash(circuit2.freeze())

        # Test that circuits can be used in sets
        circuit_set = {circuit1.freeze(), circuit2.freeze()}
        assert len(circuit_set) == 1  # Should be deduplicated

        # Test that circuits can be used as dict keys
        circuit_dict = {circuit1.freeze(): "value1", circuit2.freeze(): "value2"}
        assert len(circuit_dict) == 1  # Should be deduplicated
        assert circuit_dict[circuit1.freeze()] == "value2"

    def test_immutable_parametric_quantum_circuit_hashable(self) -> None:
        """Test that ImmutableParametricQuantumCircuit can be used in sets and
        as dict keys."""
        circuit1 = immutable_circuit()
        circuit2 = immutable_circuit()

        # Test that equal circuits have the same hash
        assert hash(circuit1) == hash(circuit2)

        # Test that circuits can be used in sets
        circuit_set = {circuit1, circuit2}
        assert len(circuit_set) == 1  # Should be deduplicated

        # Test that circuits can be used as dict keys
        circuit_dict = {circuit1: "value1", circuit2: "value2"}
        assert len(circuit_dict) == 1  # Should be deduplicated
        assert circuit_dict[circuit1] == "value2"

    def test_different_circuits_different_hashes(self) -> None:
        """Test that different parametric circuits have different hashes."""
        circuit1 = ParametricQuantumCircuit(2)
        circuit1.add_ParametricRX_gate(0)
        circuit1.add_CNOT_gate(0, 1)

        circuit2 = ParametricQuantumCircuit(2)
        circuit2.add_ParametricRY_gate(0)  # Different gate
        circuit2.add_CNOT_gate(0, 1)

        circuit3 = ParametricQuantumCircuit(3)  # Different qubit count
        circuit3.add_ParametricRX_gate(0)
        circuit3.add_CNOT_gate(0, 1)

        # Convert to immutable for hashing
        immut1 = circuit1.freeze()
        immut2 = circuit2.freeze()
        immut3 = circuit3.freeze()

        # Different circuits should have different hashes (with high probability)
        assert hash(immut1) != hash(immut2)
        assert hash(immut1) != hash(immut3)
        assert hash(immut2) != hash(immut3)

    def test_hash_stability(self) -> None:
        """Test that hash values are stable across multiple calls."""
        circuit = immutable_circuit()

        # Hash should be consistent across multiple calls
        hash1 = hash(circuit)
        hash2 = hash(circuit)
        hash3 = hash(circuit)

        assert hash1 == hash2 == hash3

    def test_hash_with_bound_vs_unbound_parameters(self) -> None:
        """Test that circuits with bound and unbound parameters behave
        correctly."""
        # Create circuit with parametric gates
        circuit = ParametricQuantumCircuit(2)
        circuit.add_X_gate(0)
        _ = circuit.add_ParametricRX_gate(0)
        _ = circuit.add_ParametricRY_gate(1)

        # Create equivalent circuit with bound parameters
        bound_circuit = ParametricQuantumCircuit(2)
        bound_circuit.add_X_gate(0)
        bound_circuit.add_RX_gate(0, 1.0)
        bound_circuit.add_RY_gate(1, 0.5)

        immut_unbound = circuit.freeze()
        immut_bound = bound_circuit.freeze()

        # Unbound and bound versions should have different hashes
        # since they have different gate structures
        assert hash(immut_unbound) != hash(immut_bound)

        # But bound parametric circuit should be consistent with itself
        bound_parametric = circuit.bind_parameters([1.0, 0.5])
        bound_parametric2 = circuit.bind_parameters([1.0, 0.5])
        assert hash(bound_parametric) == hash(bound_parametric2)

    def test_hash_with_complex_circuits(self) -> None:
        """Test hashing with more complex parametric circuits."""
        circuit1 = ParametricQuantumCircuit(4)
        circuit1.add_H_gate(0)
        circuit1.add_ParametricRX_gate(0)
        circuit1.add_CNOT_gate(0, 1)
        circuit1.add_ParametricRY_gate(1)
        circuit1.add_CZ_gate(1, 2)
        circuit1.add_ParametricRZ_gate(2)
        circuit1.add_CNOT_gate(2, 3)
        circuit1.add_ParametricPauliRotation_gate([0, 1, 2], [1, 2, 3])

        circuit2 = ParametricQuantumCircuit(4)
        circuit2.add_H_gate(0)
        circuit2.add_ParametricRX_gate(0)
        circuit2.add_CNOT_gate(0, 1)
        circuit2.add_ParametricRY_gate(1)
        circuit2.add_CZ_gate(1, 2)
        circuit2.add_ParametricRZ_gate(2)
        circuit2.add_CNOT_gate(2, 3)
        circuit2.add_ParametricPauliRotation_gate([0, 1, 2], [1, 2, 3])

        # Identical complex circuits should have same hash
        assert hash(circuit1.freeze()) == hash(circuit2.freeze())

        # Add different gate to make them different
        circuit2.add_T_gate(3)
        assert hash(circuit1.freeze()) != hash(circuit2.freeze())


# Test Rust implementation if available
try:
    from quri_parts.rust.circuit.circuit_parametric import (
        ImmutableParametricQuantumCircuit as RustImmutableParametricQuantumCircuit,
    )
    from quri_parts.rust.circuit.circuit_parametric import (
        ParametricQuantumCircuit as RustParametricQuantumCircuit,
    )

    def rust_mutable_circuit() -> RustParametricQuantumCircuit:
        circuit = RustParametricQuantumCircuit(2)
        circuit.add_X_gate(0)
        circuit.add_RX_gate(0, 1.0)
        circuit.add_CNOT_gate(0, 1)
        circuit.add_ParametricRX_gate(0)
        circuit.add_ParametricPauliRotation_gate([1], [1])
        return circuit

    def rust_immutable_circuit() -> RustImmutableParametricQuantumCircuit:
        circuit = rust_mutable_circuit()
        return RustImmutableParametricQuantumCircuit(circuit)

    class TestRustParametricQuantumCircuitHashing:
        def test_rust_parametric_circuit_hashable(self) -> None:
            """Test that Rust ParametricQuantumCircuit can be used in sets and
            as dict keys."""
            circuit1 = rust_mutable_circuit()
            circuit2 = rust_mutable_circuit()

            # Test that equal circuits have the same hash
            frozen1 = circuit1.freeze()
            frozen2 = circuit2.freeze()
            assert hash(frozen1) == hash(frozen2)

            # Test that circuits can be used in sets
            circuit_set = {frozen1, frozen2}
            assert len(circuit_set) == 1  # Should be deduplicated

            # Test that circuits can be used as dict keys
            circuit_dict = {frozen1: "value1", frozen2: "value2"}
            assert len(circuit_dict) == 1  # Should be deduplicated

        def test_rust_immutable_parametric_circuit_hashable(self) -> None:
            """Test that Rust ImmutableParametricQuantumCircuit can be used in
            sets and as dict keys."""
            circuit1 = rust_immutable_circuit()
            circuit2 = rust_immutable_circuit()

            # Test that equal circuits have the same hash
            assert hash(circuit1) == hash(circuit2)

            # Test that circuits can be used in sets
            circuit_set = {circuit1, circuit2}
            assert len(circuit_set) == 1  # Should be deduplicated

            # Test that circuits can be used as dict keys
            circuit_dict = {circuit1: "value1", circuit2: "value2"}
            assert len(circuit_dict) == 1  # Should be deduplicated

        def test_rust_different_circuits_different_hashes(self) -> None:
            """Test that different Rust parametric circuits have different
            hashes."""
            circuit1 = RustParametricQuantumCircuit(2)
            circuit1.add_ParametricRX_gate(0)
            circuit1.add_CNOT_gate(0, 1)

            circuit2 = RustParametricQuantumCircuit(2)
            circuit2.add_ParametricRY_gate(0)  # Different gate
            circuit2.add_CNOT_gate(0, 1)

            circuit3 = RustParametricQuantumCircuit(3)  # Different qubit count
            circuit3.add_ParametricRX_gate(0)
            circuit3.add_CNOT_gate(0, 1)

            # Convert to immutable for hashing
            immut1 = circuit1.freeze()
            immut2 = circuit2.freeze()
            immut3 = circuit3.freeze()

            # Different circuits should have different hashes (with high probability)
            assert hash(immut1) != hash(immut2)
            assert hash(immut1) != hash(immut3)
            assert hash(immut2) != hash(immut3)

        def test_rust_hash_stability(self) -> None:
            """Test that Rust hash values are stable across multiple calls."""
            circuit = rust_immutable_circuit()

            # Hash should be consistent across multiple calls
            hash1 = hash(circuit)
            hash2 = hash(circuit)
            hash3 = hash(circuit)

            assert hash1 == hash2 == hash3

        def test_rust_hash_with_bound_vs_unbound_parameters(self) -> None:
            """Test that Rust circuits with bound and unbound parameters behave
            correctly."""
            # Create circuit with parametric gates
            circuit = RustParametricQuantumCircuit(2)
            circuit.add_X_gate(0)
            circuit.add_ParametricRX_gate(0)
            circuit.add_ParametricRY_gate(1)

            # Create equivalent circuit with bound parameters
            bound_circuit = RustParametricQuantumCircuit(2)
            bound_circuit.add_X_gate(0)
            bound_circuit.add_RX_gate(0, 1.0)
            bound_circuit.add_RY_gate(1, 0.5)

            immut_unbound = circuit.freeze()
            immut_bound = bound_circuit.freeze()

            # Unbound and bound versions should have different hashes
            # since they have different gate structures (ParametricRX vs RX)
            assert hash(immut_unbound) != hash(immut_bound)

            # But bound parametric circuit should be consistent with itself
            bound_parametric = circuit.bind_parameters([1.0, 0.5])
            bound_parametric2 = circuit.bind_parameters([1.0, 0.5])
            assert hash(bound_parametric) == hash(bound_parametric2)

        def test_rust_complex_circuit_hashing(self) -> None:
            """Test hashing with complex Rust parametric circuits."""
            circuit1 = RustParametricQuantumCircuit(4)
            circuit1.add_H_gate(0)
            circuit1.add_ParametricRX_gate(0)
            circuit1.add_CNOT_gate(0, 1)
            circuit1.add_ParametricRY_gate(1)
            circuit1.add_CZ_gate(1, 2)
            circuit1.add_ParametricRZ_gate(2)
            circuit1.add_CNOT_gate(2, 3)
            circuit1.add_ParametricPauliRotation_gate([0, 1, 2], [1, 2, 3])

            circuit2 = RustParametricQuantumCircuit(4)
            circuit2.add_H_gate(0)
            circuit2.add_ParametricRX_gate(0)
            circuit2.add_CNOT_gate(0, 1)
            circuit2.add_ParametricRY_gate(1)
            circuit2.add_CZ_gate(1, 2)
            circuit2.add_ParametricRZ_gate(2)
            circuit2.add_CNOT_gate(2, 3)
            circuit2.add_ParametricPauliRotation_gate([0, 1, 2], [1, 2, 3])

            # Identical complex circuits should have same hash
            assert hash(circuit1.freeze()) == hash(circuit2.freeze())

            # Add different gate to make them different
            circuit2.add_T_gate(3)
            assert hash(circuit1.freeze()) != hash(circuit2.freeze())

except ImportError:
    # Rust implementation not available, skip these tests
    pass
