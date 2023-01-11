# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import networkx as nx
import pytest

from quri_parts.braket.backend import device_connectivity_graph


class MockConnectivity:
    def __init__(self, fullyConnected: bool, graph: dict[str, list[str]]) -> None:
        self.fullyConnected = fullyConnected
        self.connectivityGraph = graph


class MockParadigm:
    def __init__(self, connectivity: MockConnectivity, qubitCount: int) -> None:
        self.connectivity = connectivity
        self.qubitCount = qubitCount


class MockProperties:
    def __init__(self, paradigm: MockParadigm) -> None:
        self.paradigm = paradigm


class MockHardware:
    def __init__(self, properties: MockProperties) -> None:
        self.properties = properties


class TestDeviceConnectivity:
    def test_adjacency(self) -> None:
        graph = {"0": ["1"], "1": ["0", "2"], "2": ["1", "3"], "3": ["2"]}
        connectivity = MockConnectivity(fullyConnected=False, graph=graph)
        paradigm = MockParadigm(connectivity, qubitCount=4)
        properties = MockProperties(paradigm)
        device = MockHardware(properties)

        received_graph = device_connectivity_graph(device)

        edge_list = [(0, 1), (1, 2), (2, 3)]
        lines = [f"{a} {b}" for (a, b) in edge_list]

        constructed_graph = nx.parse_edgelist(lines, nodetype=int)

        print(received_graph.edges, constructed_graph.edges)
        assert nx.is_isomorphic(received_graph, constructed_graph)

    def test_fully_connected_graph(self) -> None:
        connectivity = MockConnectivity(fullyConnected=True, graph={})
        paradigm = MockParadigm(connectivity, qubitCount=3)
        properties = MockProperties(paradigm)
        device = MockHardware(properties)

        received_graph = device_connectivity_graph(device)
        edge_list = [(0, 1), (1, 2), (2, 0)]
        lines = [f"{a} {b}" for (a, b) in edge_list]

        constructed_graph = nx.parse_edgelist(lines, nodetype=int)

        print(received_graph.edges, constructed_graph.edges)
        assert nx.is_isomorphic(received_graph, constructed_graph)

    def test_no_connectivity_graph(self) -> None:
        class NoGraph(MockConnectivity):
            def __init__(self, fullyConnected: bool) -> None:
                self.fullyConnected = fullyConnected

        connectivity = NoGraph(fullyConnected=False)
        paradigm = MockParadigm(connectivity, qubitCount=-1)
        properties = MockProperties(paradigm)
        device = MockHardware(properties)

        with pytest.raises(ValueError, match="does not have connectivity graph"):
            device_connectivity_graph(device)

    def test_no_connectivity(self) -> None:
        class NoConnectivityParadigm(MockParadigm):
            def __init__(self, qubitCount: int) -> None:
                self.qubitCount = qubitCount

        paradigm = NoConnectivityParadigm(qubitCount=3)
        properties = MockProperties(paradigm)
        device = MockHardware(properties)

        with pytest.raises(ValueError, match="does not have connectivity attribute"):
            device_connectivity_graph(device)
