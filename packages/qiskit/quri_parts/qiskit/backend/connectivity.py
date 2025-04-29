# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import networkx as nx
from qiskit.providers import BackendV1, BackendV2


def device_connectivity_graph(device: Union[BackendV1, BackendV2]) -> nx.Graph:
    if isinstance(device, BackendV1):
        config = device.configuration()
        coupling_map = getattr(config, "coupling_map", None)

        if coupling_map is None:
            adj_list = None
        else:
            adj_list = coupling_map.get_edges()

        num_qubits = getattr(config, "num_qubits", -1)

    else:
        coupling_map = device.coupling_map
        adj_list = coupling_map.get_edges()
        num_qubits = device.num_qubits

    if adj_list is None:
        raise ValueError("Given device does not have a coupling map.")

    if num_qubits == -1:
        raise ValueError("Number of qubits not specified by the backend.")

    lines = [f"{a} {b}" for (a, b) in adj_list]
    lines.extend([str(a) for a in range(num_qubits)])

    return nx.parse_adjlist(lines)
