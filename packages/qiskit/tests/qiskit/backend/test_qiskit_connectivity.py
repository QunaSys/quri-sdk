# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "sAS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import networkx as nx
import pytest
from qiskit.providers import BackendV1, BackendV2

from quri_parts.qiskit.backend import device_connectivity_graph


class MockCouplingMap:
    def __init__(self, edge_list: Optional[list[tuple[int, int]]]):
        self._edges = edge_list

    def get_edges(self) -> Optional[list[tuple[int, int]]]:
        return self._edges


class MockConfiguration:
    def __init__(self, coupling_map: list[tuple[int, int]], num_qubits: int) -> None:
        self.coupling_map = MockCouplingMap(coupling_map)
        self.num_qubits = num_qubits


class NoMap(MockConfiguration):
    def __init__(
        self, coupling_map: Optional[list[tuple[int, int]]], num_qubits: int
    ) -> None:
        # not setting the map.
        self.num_qubits = num_qubits


class MockBackendV1(BackendV1):  # type: ignore
    def __init__(self, configuration: MockConfiguration) -> None:
        self.config = configuration

    def configuration(self) -> MockConfiguration:
        return self.config

    def _default_options(self):  # type: ignore
        pass

    def run(self):  # type: ignore
        pass


class MockBackendV2(BackendV2):  # type: ignore
    def __init__(
        self, coupling_map: Optional[list[tuple[int, int]]], num_qubits: int
    ) -> None:
        self.__coupling_map: Optional[MockCouplingMap] = None
        self.coupling_map = MockCouplingMap(coupling_map)
        self.num_qubits = num_qubits

    @property
    def coupling_map(self) -> MockCouplingMap:
        if self.__coupling_map is None:
            raise ValueError("Coupling Map has not been initialized.")
        return self.__coupling_map

    @coupling_map.setter
    def coupling_map(self, value: MockCouplingMap) -> None:
        self.__coupling_map = value

    @property
    def num_qubits(self) -> int:
        return self.__num_qubits

    @num_qubits.setter
    def num_qubits(self, value: int) -> None:
        self.__num_qubits = value

    def target(self):  # type: ignore
        pass

    def max_circuits(self):  # type: ignore
        pass

    def _default_options(self):  # type: ignore
        pass

    def run(self):  # type: ignore
        pass


class TestQiskitConnectivityGraphV1:
    def test_adjacency(self) -> None:
        coupling_map = [(0, 1), (1, 0)]
        num_qubits = 2

        config = MockConfiguration(coupling_map, num_qubits)
        backend = MockBackendV1(config)

        received_graph = device_connectivity_graph(backend)

        constructed_graph = nx.parse_edgelist(["0 1"], nodetype=int)

        assert nx.is_isomorphic(received_graph, constructed_graph)

    def test_no_coupling_map(self) -> None:
        coupling_map = [(0, 1), (1, 0)]
        num_qubits = 2

        config = NoMap(coupling_map, num_qubits)
        backend = MockBackendV1(config)

        with pytest.raises(ValueError, match="does not have a coupling map."):
            device_connectivity_graph(backend)


class TestQiskitConnectivityGraphV2:
    def test_adjacency(self) -> None:
        coupling_map = [(0, 1), (1, 0)]
        num_qubits = 2

        backend = MockBackendV2(coupling_map, num_qubits)

        received_graph = device_connectivity_graph(backend)

        constructed_graph = nx.parse_edgelist(["0 1"], nodetype=int)

        assert nx.is_isomorphic(received_graph, constructed_graph)

    def test_no_coupling_map(self) -> None:
        num_qubits = 2
        backend = MockBackendV2(coupling_map=None, num_qubits=num_qubits)

        with pytest.raises(ValueError, match="does not have a coupling map."):
            device_connectivity_graph(backend)
