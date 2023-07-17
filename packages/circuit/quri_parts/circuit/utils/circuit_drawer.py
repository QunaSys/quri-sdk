# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

# The implementation is originally from qulacs-visualizer
# https://github.com/Qulacs-Osaka/qulacs-visualizer
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt

from quri_parts.circuit import (
    NonParametricQuantumCircuit,
    ParametricQuantumGate,
    QuantumGate,
    UnboundParametricQuantumCircuitProtocol,
    gate_names,
)

_GATE_STR_MAP = {
    gate_names.X: " X ",
    gate_names.Y: " Y ",
    gate_names.Z: " Z ",
    gate_names.H: " H ",
    gate_names.S: " S ",
    gate_names.Sdag: "Sdg",
    gate_names.SqrtX: "sqX",
    gate_names.SqrtXdag: "sXd",
    gate_names.SqrtY: "sqY",
    gate_names.SqrtYdag: "sYd",
    gate_names.T: " T ",
    gate_names.Tdag: "Tdg",
    gate_names.RX: "RX ",
    gate_names.RY: "RY ",
    gate_names.RZ: "RZ ",
    gate_names.U1: "U1 ",
    gate_names.U2: "U2 ",
    gate_names.U3: "U3 ",
    gate_names.CNOT: "CX ",
    gate_names.CZ: "CZ ",
    gate_names.TOFFOLI: "TOF",
    gate_names.Pauli: " P ",
    gate_names.PauliRotation: "PR ",
    gate_names.UnitaryMatrix: "Mat",
    gate_names.ParametricRX: "PRX",
    gate_names.ParametricRY: "PRY",
    gate_names.ParametricRZ: "PRZ",
    gate_names.ParametricPauliRotation: "PPR",
}

# Width of a gate
_GATE_WIDTH = 7


def draw_circuit(
    circuit: Union[
        NonParametricQuantumCircuit, UnboundParametricQuantumCircuitProtocol
    ],
    line_length: int = 80,
) -> None:
    """Circuit drawer which outputs given circuit as an ASCII art to standard
    streams.

    Args:
        circuit: Circuit to be output.
        line_length: Maximum output line length.
    """

    qubit_count = circuit.qubit_count
    depth = circuit.depth  # depth of the circuit

    # 2D Boolean array. True if [i,j] is empty.
    gate_map = np.full((qubit_count, depth), True)

    # 4 for each qubit.
    vertical_size = qubit_count * 4

    # List of 2D str arrays to be printed.
    # circuit_pictures[i] corresponds to the circuit block at layer i.
    circuit_pictures = [
        np.full((vertical_size, _GATE_WIDTH), " ") for _ in range(depth)
    ]

    # "-" at the beginning and end of each block.
    for i in range(depth):
        for j in range(qubit_count):
            row_idx = (j + 1) * 4 - 2
            circuit_pictures[i][row_idx][0] = "-"
            circuit_pictures[i][row_idx][-1] = "-"

    # Add gate AA to ``circuit_pictures``.
    for idx, gate in enumerate(circuit.gates):
        gate_aa = _generate_gate_aa(gate, idx)
        layer, row_idx, gate_map = _place_check(gate, depth, gate_map)
        circuit_pictures[layer] = _write_gate_string(
            gate_aa, row_idx, circuit_pictures[layer]
        )

    # Join ``circuit_pictures`` into one numpy array of strings.
    interm_layer = _interm_layer(qubit_count)
    circuit_picture = interm_layer
    for i, pict in enumerate(circuit_pictures):
        circuit_picture = np.concatenate([circuit_picture, pict], axis=1)
        if i != len(circuit_pictures) - 1:
            circuit_picture = np.concatenate([circuit_picture, interm_layer], axis=1)

    # Fill in the blanks.
    circuit_picture = _connect_wire(circuit_picture)

    # Divide and fold.
    if circuit_picture.shape[1] > line_length:
        n_divs = np.ceil(circuit_picture.shape[1] / line_length).astype("int")
        output = circuit_picture[:, :line_length]
        for i in range(1, n_divs):
            output = np.concatenate(
                [
                    output,
                    np.array(
                        [
                            [""] * line_length,
                            [""] * line_length,
                            ["="] * line_length,
                            [""] * line_length,
                        ]
                    ),
                ],
                axis=0,
            )

            # Expand the last line in axis=1 to ``line_length'' and append it to
            # the end of ``output``
            if i == n_divs - 1 and circuit_picture.shape[1] % line_length:
                output = np.concatenate(
                    [
                        output,
                        np.concatenate(
                            [
                                circuit_picture[
                                    :, line_length * (n_divs - 1) :  # noqa
                                ],
                                np.full(
                                    (
                                        circuit_picture.shape[0],
                                        line_length
                                        - circuit_picture.shape[1] % line_length,
                                    ),
                                    "",
                                ),
                            ],
                            axis=1,
                        ),
                    ],
                    axis=0,
                )
            else:
                output = np.concatenate(
                    [
                        output,
                        circuit_picture[
                            :, line_length * i : line_length * (i + 1)  # noqa
                        ],
                    ],
                    axis=0,
                )

    else:
        output = circuit_picture

    for line in output:
        print("".join(line))


def _generate_gate_aa(
    gate: Union[ParametricQuantumGate, QuantumGate], gate_idx: int
) -> Sequence[str]:
    if gate_idx > 999:
        warnings.warn(
            f"Gate index larger than 999 is not supported. Index {gate_idx}"
            " is modified to 999.",
            Warning,
        )
    t_idxs = sorted(gate.target_indices)
    c_idxs = sorted(gate.control_indices)
    gate_string: list[str] = []

    if gate.name == gate_names.SWAP:
        gate_string = _create_swap_string(gate.target_indices, gate_idx)
    else:
        # Checks if control bit index < target bit index.
        if len(c_idxs) != 0 and min(t_idxs) > min(c_idxs):
            is_target_top = False
            gate_string = _create_upper_control_part(gate_string, c_idxs, t_idxs)
        else:
            is_target_top = True

        # Generate target part.
        if is_target_top:
            gate_head = "  ___  "
        else:
            gate_head = "  _|_  "
        g_str = _GATE_STR_MAP.get(gate.name, "UDF").ljust(3)
        gate_name_part = f" |{g_str}| "
        gate_body_with_wire = "-|   |-"
        gate_body = " |   | "
        gate_bottom = " |___| "

        gate_string.append(gate_head)
        gate_string.append(gate_name_part)
        g_idx_str = str(gate_idx).ljust(3)
        gate_string.append(f"-|{g_idx_str}|-")

        if len(t_idxs) == 1:
            gate_string.append(gate_bottom)
        else:
            if min(t_idxs) + 1 in t_idxs:
                gate_string.append(gate_body)
            else:
                gate_string.append(" |_ _| ")

            for q_idx in range(min(t_idxs) + 1, max(t_idxs) + 1):
                if q_idx in t_idxs:
                    # Construct top part
                    if q_idx - 1 in t_idxs:
                        gate_string.append(gate_body)
                    else:
                        gate_string.append(" _| |_ ")
                    # Construct middle part
                    gate_string.append(gate_body)
                    gate_string.append(gate_body_with_wire)
                    # Construct bottom part
                    if q_idx + 1 in t_idxs:
                        gate_string.append(gate_body)
                    else:
                        gate_string.append(gate_bottom)
                else:
                    gate_string.append("  | |  ")
                    gate_string.append("  | |  ")
                    gate_string.append("--| |--")
                    gate_string.append("  | |  ")

        if len(c_idxs) != 0 and max(t_idxs) < max(c_idxs):
            gate_string = _create_lower_control_part(gate_string, c_idxs, t_idxs)
    return gate_string


def _create_upper_control_part(
    gate_string: list[str], control_idxs: Sequence[int], target_idxs: Sequence[int]
) -> list[str]:
    ctrl_q_head = "       "
    ctrl_q_name = "       "
    ctrl_q_body = "   ●   "
    vertical_wire = "   |   "

    upper_ctrl_q_list: list[int] = [
        ctrl_idx for ctrl_idx in control_idxs if ctrl_idx < min(target_idxs)
    ]

    gate_string.append(ctrl_q_head)
    gate_string.append(ctrl_q_name)
    gate_string.append(ctrl_q_body)

    dist = min(target_idxs) - min(upper_ctrl_q_list)
    for _ in range(dist * 4 - 3):
        gate_string.append(vertical_wire)

    for i in upper_ctrl_q_list[1:]:
        dist = min(target_idxs) - i
        p = dist * 4 - 2
        gate_string[-p] = ctrl_q_body

    return gate_string


def _create_lower_control_part(
    gate_string: list[str], control_idxs: Sequence[int], target_idxs: Sequence[int]
) -> list[str]:
    control_q_body = "   ●   "
    vertical_wire = "   |   "

    lower_ctrl_q_list: list[int] = [
        idx for idx in control_idxs if idx > max(target_idxs)
    ]

    dist = max(lower_ctrl_q_list) - max(target_idxs)
    loop = dist * 4 - 1
    for _ in range(loop):
        gate_string.append(vertical_wire)

    for i in lower_ctrl_q_list:
        diff = i - max(target_idxs)
        p = diff * 4 - loop - 2
        gate_string[p] = control_q_body
    return gate_string


def _create_swap_string(target_indices: Sequence[int], gate_idx: int) -> list[str]:
    gate_string: list[str] = []
    gate_size = max(target_indices) - min(target_indices) + 1

    gate_head = "       "
    gate_index_str = str(gate_idx).ljust(3)
    gate_name_part = f"  {gate_index_str}  "
    target_part = "---x---"  # target qubits
    non_target_part = "---|---"  # non-target qubits
    gate_body = "   |   "
    gate_bottom = "       "

    gate_string.append(gate_head)
    gate_string.append(gate_name_part)
    gate_string.append(target_part)
    for i in range(1, gate_size * 4 - 4):
        if i % 4 == 0:
            gate_string.append(non_target_part)
        else:
            gate_string.append(gate_body)
    gate_string.append(target_part)
    gate_string.append(gate_bottom)
    return gate_string


def _place_check(
    gate: Union[ParametricQuantumGate, QuantumGate],
    depth: int,
    gate_map: npt.NDArray[np.bool_],
) -> tuple[int, int, npt.NDArray[np.bool_]]:
    gate_indices = (*gate.control_indices, *gate.target_indices)
    min_idx = min(gate_indices)
    row_idx = min_idx * 4
    for i in reversed(range(depth)):
        if not all(gate_map[gate_indices, i]):  # noqa
            gate_map[gate_indices, i + 1] = False  # noqa
            return i + 1, row_idx, gate_map

    gate_map[gate_indices, 0] = False
    return 0, row_idx, gate_map


def _write_gate_string(
    gate_string: Sequence[str],
    row_idx: int,
    circuit_picture: npt.NDArray[np.string_],
) -> npt.NDArray[np.string_]:
    rows, cols = circuit_picture.shape

    # Checks if existing circuit block can accomodate ``gate_string``.
    for i in range(0, circuit_picture.shape[1], _GATE_WIDTH + 1):
        if np.all(
            (
                circuit_picture[
                    row_idx : row_idx + len(gate_string), 0:_GATE_WIDTH  # noqa
                ]
                == " "
            )
            | (
                circuit_picture[
                    row_idx : row_idx + len(gate_string), 0:_GATE_WIDTH  # noqa
                ]
                == "-"
            )
        ):
            for j, line in enumerate(gate_string):
                circuit_picture[row_idx + j][i : i + _GATE_WIDTH] = list(line)  # noqa
            return circuit_picture

    # Expand ``circuit_picture``.
    interm_layer = _interm_layer(circuit_picture.shape[0] // 4)
    circuit_picture = np.concatenate([circuit_picture, interm_layer], axis=1)
    add_block = np.full((rows, 7), " ")
    for i in range(2, rows, 4):
        add_block[i][0] = "-"
        add_block[i][-1] = "-"
    circuit_picture = np.concatenate([circuit_picture, add_block], axis=1)

    for i, line in enumerate(gate_string):
        circuit_picture[row_idx + i][cols : cols + _GATE_WIDTH] = list(line)  # noqa
    return circuit_picture


def _connect_wire(circuit_picture: npt.NDArray[np.string_]) -> npt.NDArray[np.string_]:
    horizontal_size = circuit_picture.shape[1]
    for row in range(2, circuit_picture.shape[0], 4):
        p = 0
        while True:
            char_now = circuit_picture[row][p]
            if char_now == "●":
                circuit_picture[row][p + 1] = "-"
            elif char_now == "-":
                if circuit_picture[row][p + 1] == " ":
                    circuit_picture[row][p + 1] = "-"
                elif circuit_picture[row][p + 1] == "|":
                    if p + 5 < horizontal_size and circuit_picture[row][p + 5] == "|":
                        p += 5
                    elif circuit_picture[row][p + 3] == "|":
                        p += 3
                    else:
                        circuit_picture[row][p + 2] = "-"
                        p += 1
            p += 1
            if p + 1 == horizontal_size:
                break
    return circuit_picture


def _interm_layer(qubit_count: int) -> npt.NDArray[np.string_]:
    arr = np.full((qubit_count * 4, 1), " ")
    for i in range(qubit_count):
        row_idx = (i + 1) * 4 - 2
        arr[row_idx][0] = "-"
    return arr
