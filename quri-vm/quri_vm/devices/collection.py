from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from quri_parts.backend.device import DeviceProperty


class DeviceType(Enum):
    Superconducting = auto()
    Iontrap = auto()
    Neutralatom = auto()


@dataclass
class DeviceMetadata:
    system: str
    processor: str
    provider: str
    year: int
    realized: bool
    qec: bool
    device_type: DeviceType
    device_property: DeviceProperty


def chronological_plot(
    devices: Sequence[DeviceMetadata],
    extract_function: Callable[[DeviceMetadata], float],
    ylabel: str = "",
    title: str = "",
    size_inches: tuple[float, float] | None = None,
) -> Figure:
    fig, ax = plt.subplots()
    if size_inches is not None:
        fig.set_size_inches(size_inches)
    ax.set_title(title)
    ax.set_xlabel("year")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")

    color_map = plt.cm.tab10  # type: ignore
    provider_color_map: dict[str, Sequence[float]] = {}
    for d in devices:
        if d.provider in provider_color_map:
            label = None
        else:
            label = d.provider
            provider_color_map[d.provider] = color_map(len(provider_color_map))
        y = extract_function(d)
        ax.scatter(d.year, y, color=provider_color_map[d.provider], label=label)
        real = "" if d.realized else "*"
        ax.annotate(f"{d.system}{real}", (d.year + 0.2, y))
    ax.legend()
    return fig
