from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path


@dataclass(frozen=True)
class TopologySpec:
    required_qubits: int
    slack_factor: float = 1.2
    magic_factory_count: int = 4

    @property
    def target_capacity(self) -> int:
        return ceil(self.required_qubits * self.slack_factor)

    @property
    def required_sites(self) -> int:
        return self.target_capacity + self.magic_factory_count

    @property
    def width(self) -> int:
        return ceil(sqrt(self.required_sites))

    @property
    def height(self) -> int:
        return max(self.magic_factory_count, ceil(self.required_sites / self.width))


def generate_plane_topology_yaml(
    required_qubits: int,
    *,
    slack_factor: float = 1.2,
    magic_factory_count: int = 4,
) -> str:
    spec = TopologySpec(
        required_qubits=max(1, required_qubits),
        slack_factor=slack_factor,
        magic_factory_count=magic_factory_count,
    )

    lines = [
        "grids:",
        "  - type: plane",
        f"    coord: [{spec.width}, {spec.height}, 0]",
        "    magic_factory:",
    ]
    for symbol in range(spec.magic_factory_count):
        lines.extend(
            [
                f"      - symbol: {symbol}",
                f"        coord: [0, {symbol}]",
            ]
        )
    return "\n".join(lines) + "\n"


def write_generated_topology(
    path: str | Path,
    required_qubits: int,
    *,
    slack_factor: float = 1.2,
    magic_factory_count: int = 4,
) -> Path:
    path = Path(path)
    path.write_text(
        generate_plane_topology_yaml(
            required_qubits,
            slack_factor=slack_factor,
            magic_factory_count=magic_factory_count,
        )
    )
    return path
