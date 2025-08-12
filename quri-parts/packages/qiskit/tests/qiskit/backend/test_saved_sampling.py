# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any
from unittest.mock import MagicMock, Mock, patch

from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit import qasm3, transpile
from qiskit.primitives import PrimitiveResult, PubResult
from qiskit.providers.basic_provider import BasicProvider
from qiskit_ibm_runtime.runtime_job import JobStatus, RuntimeJob

from quri_parts.backend import CompositeSamplingResult
from quri_parts.circuit import QuantumCircuit
from quri_parts.qiskit.backend import (
    QiskitSavedDataSamplingBackend,
    QiskitSavedDataSamplingJob,
    QiskitSavedDataSamplingResult,
)
from quri_parts.qiskit.circuit import convert_circuit

from .mock.ibm_runtime_service_mock import mock_get_backend


def fake_run(**kwargs) -> RuntimeJob:  # type: ignore
    jjob = MagicMock(spec=RuntimeJob)

    def _always_false() -> bool:
        return False

    def _status() -> JobStatus:
        return JobStatus.DONE

    def _job_id() -> str:
        return "aaa"

    def result() -> PrimitiveResult:
        pubres = Mock(spec=PubResult)
        pubres.data.meas.get_counts.return_value = {"000001": 10}
        return PrimitiveResult([pubres])

    jjob.status = _status
    jjob.running = _always_false
    jjob.job_id = _job_id
    jjob.result = result

    return jjob


def fake_qiskit_transpile(*args: Any, **kwargs: Any) -> QiskitQuantumCircuit:
    circuit = QiskitQuantumCircuit(2)
    circuit.measure_all(inplace=True)
    return circuit


def test_qiskit_saved_data_result() -> None:
    raw_data = {
        "0110": 10238,
        "1011": 10388,
        "1110": 10254,
        "0111": 10342,
        "1010": 10203,
        "0011": 10179,
        "0001": 115224,
        "1100": 114515,
        "0100": 114650,
        "1000": 114347,
        "1001": 114832,
        "0010": 10255,
        "0000": 115088,
        "0101": 114819,
        "1111": 10157,
        "1101": 114509,
    }
    expected_counter = {
        6: 10238,
        11: 10388,
        14: 10254,
        7: 10342,
        10: 10203,
        3: 10179,
        1: 115224,
        12: 114515,
        4: 114650,
        8: 114347,
        9: 114832,
        2: 10255,
        0: 115088,
        5: 114819,
        15: 10157,
        13: 114509,
    }

    saved_data_sampling_result = QiskitSavedDataSamplingResult(raw_data=raw_data)
    assert saved_data_sampling_result.counts == expected_counter


def test_qiskit_saved_data_job() -> None:
    circuit_qasm_str = """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    creg meas[4];
    u2(0,-2.9115927) q[0];
    u3(0.58079633,0,-pi) q[1];
    u2(0,-2.2715927) q[2];
    u2(-0.16,-pi) q[3];
    barrier q[0],q[1],q[2],q[3];
    measure q[0] -> meas[0];
    measure q[1] -> meas[1];
    measure q[2] -> meas[2];
    measure q[3] -> meas[3];
    """
    n_shots = 1000000
    raw_data = {
        "0110": 10238,
        "1011": 10388,
        "1110": 10254,
        "0111": 10342,
        "1010": 10203,
        "0011": 10179,
        "0001": 115224,
        "1100": 114515,
        "0100": 114650,
        "1000": 114347,
        "1001": 114832,
        "0010": 10255,
        "0000": 115088,
        "0101": 114819,
        "1111": 10157,
        "1101": 114509,
    }
    test_qiskit_saved_data_result = QiskitSavedDataSamplingResult(raw_data)
    sampling_data_sampling_job = QiskitSavedDataSamplingJob(
        circuit_qasm_str, n_shots, test_qiskit_saved_data_result
    )

    assert sampling_data_sampling_job.n_shots == 1000000
    assert sampling_data_sampling_job.saved_result.counts == {
        6: 10238,
        11: 10388,
        14: 10254,
        7: 10342,
        10: 10203,
        3: 10179,
        1: 115224,
        12: 114515,
        4: 114650,
        8: 114347,
        9: 114832,
        2: 10255,
        0: 115088,
        5: 114819,
        15: 10157,
        13: 114509,
    }


def test_replay_sampling_output_and_memory() -> None:
    # Build json string
    backend = BasicProvider().get_backend("basic_simulator")

    qp_circuit1 = QuantumCircuit(4)
    qp_circuit1.add_H_gate(0)
    qp_circuit1.add_H_gate(1)
    qp_circuit1.add_H_gate(2)
    qp_circuit1.add_H_gate(3)
    qp_circuit1.add_RX_gate(0, 0.23)
    qp_circuit1.add_RY_gate(1, -0.99)
    qp_circuit1.add_RX_gate(2, 0.87)
    qp_circuit1.add_RZ_gate(3, -0.16)

    qp_circuit2 = QuantumCircuit(4)
    qp_circuit2.add_H_gate(0)
    qp_circuit2.add_H_gate(1)
    qp_circuit2.add_H_gate(2)
    qp_circuit2.add_H_gate(3)
    qp_circuit2.add_RX_gate(0, 123)
    qp_circuit2.add_RY_gate(1, 456)
    qp_circuit2.add_RX_gate(2, 789)
    qp_circuit2.add_RZ_gate(3, -1283)

    qp_circuit3 = QuantumCircuit(4)
    qp_circuit3.add_H_gate(0)
    qp_circuit3.add_H_gate(1)
    qp_circuit3.add_H_gate(2)
    qp_circuit3.add_H_gate(3)
    qp_circuit3.add_RX_gate(0, 0.998)
    qp_circuit3.add_RY_gate(1, 1.928)
    qp_circuit3.add_RX_gate(2, -10.39)
    qp_circuit3.add_RZ_gate(3, -0.1023)

    circuit1_qasm_str = qasm3.dumps(
        transpile(convert_circuit(qp_circuit1).measure_all(inplace=False), backend)
    )

    circuit2_qasm_str = qasm3.dumps(
        transpile(convert_circuit(qp_circuit2).measure_all(inplace=False), backend)
    )

    circuit3_qasm_str = qasm3.dumps(
        transpile(convert_circuit(qp_circuit3).measure_all(inplace=False), backend)
    )

    saved_data_list = [
        {
            "circuit_qasm": circuit1_qasm_str,
            "n_shots": 1000000,
            "saved_result": {
                "raw_data": {
                    "0010": 10373,
                    "1110": 10348,
                    "1010": 10174,
                    "1011": 10307,
                    "0101": 115124,
                    "1001": 114595,
                    "1000": 114544,
                    "0100": 114598,
                    "1100": 115050,
                    "0111": 10251,
                    "0110": 10191,
                    "0011": 10172,
                    "0001": 114214,
                    "1111": 10383,
                    "1101": 114767,
                    "0000": 114909,
                }
            },
        },
        {
            "circuit_qasm": circuit1_qasm_str,
            "n_shots": 1000000,
            "saved_result": {
                "raw_data": {
                    "0110": 10238,
                    "1011": 10388,
                    "1110": 10254,
                    "0111": 10342,
                    "1010": 10203,
                    "0011": 10179,
                    "0001": 115224,
                    "1100": 114515,
                    "0100": 114650,
                    "1000": 114347,
                    "1001": 114832,
                    "0010": 10255,
                    "0000": 115088,
                    "0101": 114819,
                    "1111": 10157,
                    "1101": 114509,
                }
            },
        },
        {
            "circuit_qasm": circuit1_qasm_str,
            "n_shots": 1000000,
            "saved_result": {
                "raw_data": {
                    "0010": 10453,
                    "0110": 10131,
                    "1010": 10258,
                    "1011": 10326,
                    "0101": 114356,
                    "0000": 114440,
                    "0011": 10192,
                    "0001": 114352,
                    "1100": 114659,
                    "0100": 114908,
                    "1001": 114963,
                    "1110": 10304,
                    "1000": 114610,
                    "1101": 115473,
                    "1111": 10292,
                    "0111": 10283,
                }
            },
        },
        {
            "circuit_qasm": circuit2_qasm_str,
            "n_shots": 1000000,
            "saved_result": {
                "raw_data": {
                    "1011": 34397,
                    "1110": 34041,
                    "0110": 34715,
                    "1100": 90801,
                    "0100": 90877,
                    "0011": 34552,
                    "0001": 90504,
                    "1001": 90858,
                    "1000": 90061,
                    "1101": 90900,
                    "1111": 34405,
                    "0111": 34265,
                    "0101": 90604,
                    "0010": 34374,
                    "0000": 90357,
                    "1010": 34289,
                }
            },
        },
        {
            "circuit_qasm": circuit2_qasm_str,
            "n_shots": 1000000,
            "saved_result": {
                "raw_data": {
                    "0111": 34366,
                    "0010": 34026,
                    "1011": 34314,
                    "0110": 34091,
                    "0101": 91442,
                    "1010": 34197,
                    "1101": 90708,
                    "1111": 34314,
                    "0001": 90921,
                    "0011": 34021,
                    "1001": 90850,
                    "0100": 90964,
                    "1100": 90299,
                    "1000": 90551,
                    "1110": 34344,
                    "0000": 90592,
                }
            },
        },
        {
            "circuit_qasm": circuit2_qasm_str,
            "n_shots": 1000000,
            "saved_result": {
                "raw_data": {
                    "0111": 34336,
                    "0010": 34370,
                    "1011": 34417,
                    "0110": 34566,
                    "1010": 33991,
                    "0101": 91081,
                    "0001": 90492,
                    "0011": 34318,
                    "1101": 90390,
                    "1111": 34214,
                    "1110": 34216,
                    "0100": 90985,
                    "1100": 90528,
                    "1000": 91073,
                    "0000": 90772,
                    "1001": 90251,
                }
            },
        },
        {
            "circuit_qasm": circuit2_qasm_str,
            "n_shots": 1000000,
            "saved_result": {
                "raw_data": {
                    "1010": 34373,
                    "0010": 34790,
                    "0110": 34299,
                    "0111": 33969,
                    "1011": 34042,
                    "0101": 90691,
                    "1101": 90904,
                    "1111": 33975,
                    "0001": 90519,
                    "0011": 34348,
                    "1001": 90584,
                    "1110": 34184,
                    "1000": 90780,
                    "0100": 90973,
                    "1100": 90757,
                    "0000": 90812,
                }
            },
        },
        {
            "circuit_qasm": circuit3_qasm_str,
            "n_shots": 4000,
            "saved_result": {
                "raw_data": {
                    "0101": 15,
                    "0000": 16,
                    "1001": 10,
                    "1010": 497,
                    "1000": 13,
                    "1110": 508,
                    "1011": 464,
                    "0010": 466,
                    "0111": 494,
                    "1101": 16,
                    "1111": 492,
                    "0011": 487,
                    "0001": 17,
                    "1100": 16,
                    "0100": 14,
                    "0110": 475,
                }
            },
        },
        {
            "circuit_qasm": circuit1_qasm_str,
            "n_shots": 200000,
            "saved_result": {
                "raw_data": {
                    "0110": 2071,
                    "0111": 1990,
                    "1110": 2129,
                    "0010": 2029,
                    "1011": 2001,
                    "1010": 2059,
                    "1100": 22558,
                    "0100": 22761,
                    "0011": 2079,
                    "0001": 23159,
                    "1001": 23169,
                    "1111": 1954,
                    "1101": 23127,
                    "0000": 23012,
                    "0101": 23007,
                    "1000": 22895,
                }
            },
        },
        {
            "circuit_qasm": circuit2_qasm_str,
            "n_shots": 8000,
            "saved_result": {
                "raw_data": {
                    "1011": 259,
                    "1010": 263,
                    "1110": 271,
                    "0111": 264,
                    "1111": 264,
                    "1101": 758,
                    "0110": 281,
                    "0011": 287,
                    "0001": 768,
                    "1100": 726,
                    "0100": 688,
                    "1001": 702,
                    "1000": 712,
                    "0000": 741,
                    "0010": 272,
                    "0101": 744,
                }
            },
        },
        {
            "circuit_qasm": circuit3_qasm_str,
            "n_shots": 1000000,
            "saved_result": {
                "raw_data": {
                    "1000": 3914,
                    "0101": 3974,
                    "0000": 3903,
                    "1001": 3937,
                    "1111": 120649,
                    "1101": 3951,
                    "1010": 121043,
                    "0011": 121392,
                    "0001": 3973,
                    "0110": 120760,
                    "0100": 3929,
                    "1100": 4002,
                    "1110": 120957,
                    "0010": 121698,
                    "1011": 120918,
                    "0111": 121000,
                }
            },
        },
        {
            "circuit_qasm": circuit3_qasm_str,
            "n_shots": 500000,
            "saved_result": {
                "raw_data": {
                    "0000": 1974,
                    "0101": 2035,
                    "1001": 2030,
                    "1011": 60093,
                    "0010": 60929,
                    "1100": 2081,
                    "0100": 2045,
                    "0110": 60742,
                    "1101": 1996,
                    "1111": 60744,
                    "0001": 1991,
                    "0011": 59980,
                    "0111": 60564,
                    "1010": 60370,
                    "1000": 1930,
                    "1110": 60496,
                }
            },
        },
    ]

    saved_data_str = json.dumps(saved_data_list)

    # Set up sampling backend
    saved_data_backend = QiskitSavedDataSamplingBackend(
        backend=backend, saved_data=saved_data_str
    )

    n_shots_1 = int(2e6)
    n_shots_2 = int(4e6)
    n_shots_3 = int(4000)
    n_shots_4 = int(1.2e6)
    n_shots_5 = int(8000)
    n_shots_6 = int(1.5e6)

    # sampling experiment 1
    job_1 = saved_data_backend.sample(qp_circuit1, n_shots_1)
    result_1 = job_1.result()
    assert isinstance(result_1, CompositeSamplingResult)
    for component_result in result_1.results:
        assert isinstance(component_result, QiskitSavedDataSamplingResult)
    assert result_1.counts == {
        2: 20628,
        14: 20602,
        10: 20377,
        11: 20695,
        5: 229943,
        9: 229427,
        8: 228891,
        4: 229248,
        12: 229565,
        7: 20593,
        6: 20429,
        3: 20351,
        1: 229438,
        15: 20540,
        13: 229276,
        0: 229997,
    }

    # sampling experiment 2
    job_2 = saved_data_backend.sample(qp_circuit2, n_shots_2)
    result_2 = job_2.result()
    assert isinstance(result_2, CompositeSamplingResult)
    for component_result in result_2.results:
        assert isinstance(component_result, QiskitSavedDataSamplingResult)
    assert result_2.counts == {
        11: 137170,
        14: 136785,
        6: 137671,
        12: 362385,
        4: 363799,
        3: 137239,
        1: 362436,
        9: 362543,
        8: 362465,
        13: 362902,
        15: 136908,
        7: 136936,
        5: 363818,
        2: 137560,
        0: 362533,
        10: 136850,
    }

    # sampling experiment 3
    job_3 = saved_data_backend.sample(qp_circuit3, n_shots_3)
    result_3 = job_3.result()
    assert isinstance(result_3, QiskitSavedDataSamplingResult)
    assert result_3.counts == {
        5: 15,
        0: 16,
        9: 10,
        10: 497,
        8: 13,
        14: 508,
        11: 464,
        2: 466,
        7: 494,
        13: 16,
        15: 492,
        3: 487,
        1: 17,
        12: 16,
        4: 14,
        6: 475,
    }

    # sampling experiment 4
    job_4 = saved_data_backend.sample(qp_circuit1, n_shots_4)
    result_4 = job_4.result()
    assert isinstance(result_4, CompositeSamplingResult)
    for component_result in result_4.results:
        assert isinstance(component_result, QiskitSavedDataSamplingResult)
    assert result_4.counts == {
        2: 12482,
        6: 12202,
        10: 12317,
        11: 12327,
        5: 137363,
        0: 137452,
        3: 12271,
        1: 137511,
        12: 137217,
        4: 137669,
        9: 138132,
        14: 12433,
        8: 137505,
        13: 138600,
        15: 12246,
        7: 12273,
    }

    # sampling experiment 5
    job_5 = saved_data_backend.sample(qp_circuit2, n_shots_5)
    result_5 = job_5.result()
    assert isinstance(result_5, QiskitSavedDataSamplingResult)
    assert result_5.counts == {
        11: 259,
        10: 263,
        14: 271,
        7: 264,
        15: 264,
        13: 758,
        6: 281,
        3: 287,
        1: 768,
        12: 726,
        4: 688,
        9: 702,
        8: 712,
        0: 741,
        2: 272,
        5: 744,
    }

    # sampling experiment 6
    job_6 = saved_data_backend.sample(qp_circuit3, n_shots_6)
    result_6 = job_6.result()
    assert isinstance(result_6, CompositeSamplingResult)
    for component_result in result_6.results:
        assert isinstance(component_result, QiskitSavedDataSamplingResult)
    assert result_6.counts == {
        8: 5844,
        5: 6009,
        0: 5877,
        9: 5967,
        15: 181393,
        13: 5947,
        10: 181413,
        3: 181372,
        1: 5964,
        6: 181502,
        4: 5974,
        12: 6083,
        14: 181453,
        2: 182627,
        11: 181011,
        7: 181564,
    }


@patch("qiskit.transpile", fake_qiskit_transpile)
def test_runtime_replay_sampling_output_and_memory() -> None:
    # Build json string
    runtime_service = mock_get_backend()
    service = runtime_service()
    service.run = fake_run
    backend = service.backend()

    qp_circuit1 = QuantumCircuit(4)
    qp_circuit1.add_H_gate(0)
    qp_circuit1.add_H_gate(1)
    qp_circuit1.add_H_gate(2)
    qp_circuit1.add_H_gate(3)
    qp_circuit1.add_RX_gate(0, 0.23)
    qp_circuit1.add_RY_gate(1, -0.99)
    qp_circuit1.add_RX_gate(2, 0.87)
    qp_circuit1.add_RZ_gate(3, -0.16)

    qp_circuit2 = QuantumCircuit(4)
    qp_circuit2.add_H_gate(0)
    qp_circuit2.add_H_gate(1)
    qp_circuit2.add_H_gate(2)
    qp_circuit2.add_H_gate(3)
    qp_circuit2.add_RX_gate(0, 123)
    qp_circuit2.add_RY_gate(1, 456)
    qp_circuit2.add_RX_gate(2, 789)
    qp_circuit2.add_RZ_gate(3, -1283)

    qp_circuit3 = QuantumCircuit(4)
    qp_circuit3.add_H_gate(0)
    qp_circuit3.add_H_gate(1)
    qp_circuit3.add_H_gate(2)
    qp_circuit3.add_H_gate(3)
    qp_circuit3.add_RX_gate(0, 0.998)
    qp_circuit3.add_RY_gate(1, 1.928)
    qp_circuit3.add_RX_gate(2, -10.39)
    qp_circuit3.add_RZ_gate(3, -0.1023)

    circuit1_qasm_str = qasm3.dumps(
        fake_qiskit_transpile(
            convert_circuit(qp_circuit1).measure_all(inplace=False), backend
        )
    )

    circuit2_qasm_str = qasm3.dumps(
        fake_qiskit_transpile(
            convert_circuit(qp_circuit2).measure_all(inplace=False), backend
        )
    )

    circuit3_qasm_str = qasm3.dumps(
        fake_qiskit_transpile(
            convert_circuit(qp_circuit3).measure_all(inplace=False), backend
        )
    )

    saved_data_list = [
        {
            "circuit_qasm": circuit1_qasm_str,
            "n_shots": 10,
            "saved_result": {"raw_data": {"0001": 10}},
        },
        {
            "circuit_qasm": circuit1_qasm_str,
            "n_shots": 10,
            "saved_result": {"raw_data": {"0001": 10}},
        },
        {
            "circuit_qasm": circuit1_qasm_str,
            "n_shots": 10,
            "saved_result": {"raw_data": {"0001": 10}},
        },
        {
            "circuit_qasm": circuit2_qasm_str,
            "n_shots": 10,
            "saved_result": {"raw_data": {"0001": 10}},
        },
        {
            "circuit_qasm": circuit2_qasm_str,
            "n_shots": 10,
            "saved_result": {"raw_data": {"0001": 10}},
        },
        {
            "circuit_qasm": circuit2_qasm_str,
            "n_shots": 10,
            "saved_result": {"raw_data": {"0001": 10}},
        },
        {
            "circuit_qasm": circuit2_qasm_str,
            "n_shots": 10,
            "saved_result": {"raw_data": {"0001": 10}},
        },
        {
            "circuit_qasm": circuit3_qasm_str,
            "n_shots": 10,
            "saved_result": {"raw_data": {"0001": 10}},
        },
        {
            "circuit_qasm": circuit1_qasm_str,
            "n_shots": 10,
            "saved_result": {"raw_data": {"0001": 10}},
        },
        {
            "circuit_qasm": circuit2_qasm_str,
            "n_shots": 10,
            "saved_result": {"raw_data": {"0001": 10}},
        },
        {
            "circuit_qasm": circuit3_qasm_str,
            "n_shots": 10,
            "saved_result": {"raw_data": {"0001": 10}},
        },
        {
            "circuit_qasm": circuit3_qasm_str,
            "n_shots": 10,
            "saved_result": {"raw_data": {"0001": 10}},
        },
    ]

    saved_data_str = json.dumps(saved_data_list)

    # Set up sampling backend
    saved_data_backend = QiskitSavedDataSamplingBackend(
        backend=backend, saved_data=saved_data_str
    )
    saved_data_backend._max_shots = 10

    n_shots_1 = int(30)
    n_shots_2 = int(40)
    n_shots_3 = int(10)
    n_shots_4 = int(10)
    n_shots_5 = int(10)
    n_shots_6 = int(20)

    # sampling experiment 1
    job_1 = saved_data_backend.sample(qp_circuit1, n_shots_1)
    result_1 = job_1.result()
    assert isinstance(result_1, CompositeSamplingResult)
    for component_result in result_1.results:
        assert isinstance(component_result, QiskitSavedDataSamplingResult)
    assert result_1.counts == {1: 30.0}

    # sampling experiment 2
    job_2 = saved_data_backend.sample(qp_circuit2, n_shots_2)
    result_2 = job_2.result()
    assert isinstance(result_2, CompositeSamplingResult)
    for component_result in result_2.results:
        assert isinstance(component_result, QiskitSavedDataSamplingResult)
    assert result_2.counts == {1: 40.0}

    # sampling experiment 3
    job_3 = saved_data_backend.sample(qp_circuit3, n_shots_3)
    result_3 = job_3.result()
    assert isinstance(result_3, QiskitSavedDataSamplingResult)
    assert result_3.counts == {1: 10.0}

    # sampling experiment 4
    job_4 = saved_data_backend.sample(qp_circuit1, n_shots_4)
    result_4 = job_4.result()
    assert isinstance(result_4, QiskitSavedDataSamplingResult)
    assert result_4.counts == {1: 10.0}

    # sampling experiment 5
    job_5 = saved_data_backend.sample(qp_circuit2, n_shots_5)
    result_5 = job_5.result()
    assert isinstance(result_5, QiskitSavedDataSamplingResult)
    assert result_5.counts == {1: 10.0}

    # sampling experiment 6
    job_6 = saved_data_backend.sample(qp_circuit3, n_shots_6)
    result_6 = job_6.result()
    assert isinstance(result_6, CompositeSamplingResult)
    for component_result in result_6.results:
        assert isinstance(component_result, QiskitSavedDataSamplingResult)
    assert result_6.counts == {1: 20.0}
