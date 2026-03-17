import tempfile
from typing import Any, cast

import pytest

import pyqret.backend as backend  # type: ignore

import quri_parts.qsub.lib.std as std
from quri_parts.qret.convert_qsub import create_module_from_qsub_op
from quri_parts.qsub.lib.qpe import QPE
from quri_parts.qsub.opsub import UnitarySubDef, opsub
from quri_parts.qsub.sub import SubBuilder

OptLevel = cast(Any, getattr(backend, "OptLevel"))


class _U(UnitarySubDef):
    name = "U"
    qubit_count = 2

    def sub(self, builder: SubBuilder) -> None:
        q0, q1 = builder.qubits
        builder.add_op(std.H, (q0,))
        builder.add_op(std.RX(0.2), (q0,))
        builder.add_op(std.RZ(0.15), (q1,))
        builder.add_op(std.CNOT, (q0, q1))


U, _ = opsub(_U)

_TOPOLOGY_PLANE = """
grids:
  - type: plane
    coord: [20, 20, 0]
    magic_factory:
      - symbol: 0
        coord: [0, 0]
      - symbol: 1
        coord: [0, 1]
      - symbol: 2
        coord: [0, 2]
      - symbol: 3
        coord: [0, 3]
"""


class TestPyQRETFeatures:
    def _create_ftqc_like_option(self, topology: str = "") -> Any:
        if hasattr(backend, "FTQCOption"):
            return backend.FTQCOption(topology=topology)
        return backend.ScLsFixedV0Option(topology=topology)

    def _compile_option_kwargs(self, option: Any) -> dict[str, Any]:
        if hasattr(backend, "FTQCOption"):
            return {"ftqc_option": option}
        return {"sc_ls_fixed_v0_option": option}

    def _compile_sub_option(self, option: Any) -> Any:
        if hasattr(backend, "FTQCOption"):
            return option.ftqc_option
        return option.sc_ls_fixed_v0_option

    def _write_topology_file(self) -> str:
        handle = tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml")
        handle.write(_TOPOLOGY_PLANE)
        handle.flush()
        return handle.name

    def test_backend_compile_option(self) -> None:
        option = backend.CompileOption(OptLevel.O0)

        assert option.opt_level == OptLevel.O0
        assert option.verbose is False
        with pytest.raises(RuntimeError):
            _ = self._compile_sub_option(option)

    def test_backend_ftqc_option_defaults(self) -> None:
        option = self._create_ftqc_like_option()

        assert str(option.topology) == ""
        assert option.magic_generation_period == 15
        assert option.maximum_magic_state_stock == 10000
        assert option.entanglement_generation_period == 100
        assert option.maximum_entangled_state_stock == 10
        assert option.reaction_time == 1

    def test_backend_compile_option_with_ftqc_option(self) -> None:
        ftqc = self._create_ftqc_like_option()
        option = backend.CompileOption(OptLevel.O0, **self._compile_option_kwargs(ftqc))

        assert self._compile_sub_option(option).topology == ftqc.topology

    def test_ir_cfg_and_call_graph(self) -> None:
        qpe_u_op = QPE(2, U)
        module = create_module_from_qsub_op(qpe_u_op)
        circuit = module.get_circuit(qpe_u_op.id.to_str())
        ir = circuit.get_ir()

        cfg = ir.gen_cfg()
        call_graph = ir.gen_call_graph(display_num_calls=True)

        assert isinstance(cfg, str) and cfg
        assert "entry" in cfg
        assert isinstance(call_graph, str) and call_graph
        assert "digraph" in call_graph

    def test_backend_resource_estimation(self) -> None:
        qpe_u_op = QPE(2, U)
        module = create_module_from_qsub_op(qpe_u_op)
        circuit = module.get_circuit(qpe_u_op.id.to_str())

        topology_path = self._write_topology_file()
        ftqc_like_option = self._create_ftqc_like_option(topology=topology_path)
        option = backend.CompileOption(
            OptLevel.O0,
            **self._compile_option_kwargs(ftqc_like_option),
        )
        compiler = backend.Compiler(option)
        compile_result = compiler.compile(circuit)
        info = compiler.get_compile_info()

        run_order = compile_result.get_run_order()
        elapsed = compile_result.get_elapsed_time()

        assert isinstance(run_order, list)
        assert isinstance(elapsed, list)
        assert len(run_order) == len(elapsed)
        assert info.gate_count > 0
        assert info.gate_depth > 0
        assert info.runtime >= 0
        assert info.qubit_volume >= 0

    def test_backend_compiler_pass_list(self) -> None:
        topology_path = self._write_topology_file()
        ftqc_like_option = self._create_ftqc_like_option(topology=topology_path)
        option = backend.CompileOption(
            OptLevel.O0,
            **self._compile_option_kwargs(ftqc_like_option),
        )
        compiler = backend.Compiler(option)

        available = compiler.get_available_passes()
        assert isinstance(available, list)
        assert available
