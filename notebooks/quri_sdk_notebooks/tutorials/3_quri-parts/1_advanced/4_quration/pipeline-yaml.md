# Pipeline YAML Reference

**⚠️This document is genrated by Claude Code**

Used with `qret compile --pipeline <file.yaml>` and `qret opt --pipeline <file.yaml>`.

---

## Top-Level Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input` | string | — | Path to input file |
| `output` | string | `a.json` | Path to output file |
| `source` | string | `IR` | Source format: `IR`, `OpenQASM2`, or `SC_LS_FIXED_V0` |
| `target` | string | `SC_LS_FIXED_V0` | Target machine name. See [Targets](#targets) below |
| `function` | string | — | Function name to compile (required when source is `IR`) |
| `sc_ls_fixed_v0_topology` | string | — | Path to topology YAML/JSON file |
| `sc_ls_fixed_v0_machine_type` | string | `auto` | See [Machine Types](#machine-types-sc_ls_fixed_v0_machine_type) below |
| `sc_ls_fixed_v0_pass` | sequence | — | Ordered list of passes to run (see below) |
| `pass` | sequence | — | IR-level passes for `qret opt` |

---

## Passes

### SC_LS_FIXED_V0 Passes (`sc_ls_fixed_v0_pass`)

| Pass Name | Description |
|-----------|-------------|
| `sc_ls_fixed_v0::init_compile_info` | Initialize compilation metadata (run first) |
| `sc_ls_fixed_v0::mapping` | Map logical qubits to physical positions on topology |
| `sc_ls_fixed_v0::routing` | Route instructions through the lattice |
| `sc_ls_fixed_v0::calc_info_without_topology` | Compute stats independent of topology (beat count, etc.) |
| `sc_ls_fixed_v0::calc_info_with_topology` | Compute stats dependent on topology (physical qubit count, etc.) |
| `sc_ls_fixed_v0::calc_info_with_qec_resource_estimation` | QEC resource estimation (physical qubits, time) |
| `sc_ls_fixed_v0::dump_compile_info` | Output compilation summary |
| `sc_ls_fixed_v0::runtime_simulation_pruning` | Prune branches via runtime simulation |
| `sc_ls_fixed_v0::external` | Run an external pass |

### IR Passes (`pass`, for `qret opt`)

| Pass Name | Description |
|-----------|-------------|
| `ir::delete_opt_hint` | Remove optimization hints |
| `ir::inliner` | Inline function calls |
| `ir::recursive_inliner` | Recursively inline all functions |
| `ir::decompose_inst` | Decompose instructions |
| `ir::ignore_global_phase` | Remove global phase tracking |
| `ir::static_condition_pruning` | Prune statically deterministic branches |
| `ir::external` | Run an external pass |

### External Pass Format

```yaml
sc_ls_fixed_v0_pass:
  - name: MyPass
    cmd: python3 my_pass.py in.json out.json
    input: in.json
    output: out.json
```

---

## Machine Types (`sc_ls_fixed_v0_machine_type`)

| Value | Description |
|-------|-------------|
| `auto` | (default) Infer the minimum required type from the topology file |
| `Dim2` | Single-layer 2D array of surface codes. Standard single-chip architecture. Supports lattice surgery, magic factories, MOVE, CNOT. Required for PBC mode. |
| `Dim3` | Multiple stacked 2D layers. Adds transversal gates between layers (`CNOT_TRANS`, `SWAP_TRANS`, `MOVE_TRANS`) which are faster than lattice surgery CNOTs. Topology must have multiple z-layers. |
| `DistributedDim2` | Multiple separate 2D chips connected via entanglement factories. Each chip is a Dim2 node; cross-chip operations use `LATTICE_SURGERY_MULTINODE` and `MOVE_ENTANGLEMENT`. Requires entanglement factory pairs (`E`) in topology. |
| `DistributedDim3` | Distributed multi-chip with Dim3 nodes. **Currently unsupported** — will error at compile time. |

### Compatibility

When specifying `sc_ls_fixed_v0_machine_type` manually, it must be compatible with what the topology file implies:

- `Dim2` topology → can use `Dim2` only
- `Dim3` topology → can use `Dim3` only
- `DistributedDim2` topology → can use `DistributedDim2` only
- A more capable type cannot be applied to a less capable topology

If incompatible, compilation will error. Use `auto` to let the compiler infer the correct type.

---

## SC_LS_FIXED_V0 Options

### Magic State Factory

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sc_ls_fixed_v0_use_magic_state_cultivation` | flag | false | Use magic state cultivation method |
| `sc_ls_fixed_v0_magic_factory_seed_offset` | uint64 | 0 | RNG seed offset for factory |
| `sc_ls_fixed_v0_magic_generation_period` | uint64 | 15 | Beats per magic state |
| `sc_ls_fixed_v0_prob_magic_state_creation` | double | 1.0 | Success probability per attempt |
| `sc_ls_fixed_v0_maximum_magic_state_stock` | uint64 | 10000 | Max storable magic states |

### Entanglement Factory (Distributed)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sc_ls_fixed_v0_entanglement_generation_period` | uint64 | 100 | Beats per entangled pair |
| `sc_ls_fixed_v0_maximum_entangled_state_stock` | uint64 | 10 | Max storable entangled pairs |

### System

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sc_ls_fixed_v0_reaction_time` | uint64 | 1 | Feed-forward latency in beats (measurement → correction) |
| `sc_ls_fixed_v0_enable_pbc_mode` | flag | false | Enable Pauli Based Computing mode (Dim2 only; all qubits must be measured) |

### Output

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sc_ls_fixed_v0_dump_compile_info_to_json` | string | `""` | Write compilation stats to JSON |
| `sc_ls_fixed_v0_dump_compile_info_to_markdown` | string | `""` | Write compilation stats to Markdown |
| `sc_ls_fixed_v0_dump_pbc_string` | string | `""` | Write Pauli basis string (PBC mode only) |
| `sc_ls_fixed_v0-print-inst-metadata` | bool | false | Include beat/coordinate metadata in output JSON |

### QEC Resource Estimation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sc_ls_fixed_v0_physical_error_rate` | double | 0.0 | Physical error rate (p) |
| `sc_ls_fixed_v0_drop_rate` | double | 0.0 | Photon drop rate (Λ) |
| `sc_ls_fixed_v0_code_cycle_time_sec` | double | 0.0 | Code cycle time in seconds |
| `sc_ls_fixed_v0_allowed_failure_prob` | double | 0.0 | Allowed logical failure probability (ε) |

### Advanced / Hidden

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sc_ls_fixed_v0-mapping-algorithm` | uint32 | 1 | `0`: topology-file-based, `1`: auto |
| `sc_ls_fixed_v0-partition-algorithm` | uint32 | 0 | `0`: Greedy, `1`: Random, `2`: METIS (unimplemented) |
| `sc_ls_fixed_v0-partition-seed` | uint64 | 314 | RNG seed for Random partition |
| `sc_ls_fixed_v0-find-place-algorithm` | uint32 | 0 | `0`: EnoughSpaceSoft (allow boundary), `1`: EnoughSpaceHard |
| `sc_ls_fixed_v0-inst-queue-weight-algorithm` | int32 | 2 | Instruction scheduling priority: `0`: index, `1`: type, `2`: InvDepth |
| `sc_ls_fixed_v0-inst-queue-peek-size` | uint64 | 1000 | Instructions to look ahead in queue |
| `sc_ls_fixed_v0-state-buffer-width` | uint64 | 20 | Quantum state buffer width in beats |
| `sc_ls_fixed_v0_runtime_simulation_pruning_seed` | uint64 | 0 | Seed for runtime simulation pruning |
| `ir-static-condition-pruning-seed` | uint64 | 0 | Seed for IR static condition pruning |

---

## Minimal Example

```yaml
source: IR
input: circuit.json
function: MyFunction
output: out.json
sc_ls_fixed_v0_topology: topology.yaml
sc_ls_fixed_v0_machine_type: Dim2
sc_ls_fixed_v0_magic_generation_period: 15
sc_ls_fixed_v0_maximum_magic_state_stock: 10000
sc_ls_fixed_v0_entanglement_generation_period: 100
sc_ls_fixed_v0_maximum_entangled_state_stock: 10
sc_ls_fixed_v0_reaction_time: 1
sc_ls_fixed_v0_pass:
  - sc_ls_fixed_v0::init_compile_info
  - sc_ls_fixed_v0::mapping
  - sc_ls_fixed_v0::routing
  - sc_ls_fixed_v0::calc_info_without_topology
  - sc_ls_fixed_v0::calc_info_with_topology
  - sc_ls_fixed_v0::dump_compile_info
```

See [quration-core/examples/data/pipeline/](../quration-core/examples/data/pipeline/) for more examples.
