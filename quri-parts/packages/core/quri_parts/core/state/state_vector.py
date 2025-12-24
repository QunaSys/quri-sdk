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
from abc import ABC
from math import comb
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from quri_parts.circuit.circuit import GateSequence, ImmutableQuantumCircuit

from ..utils.array import readonly_array
from .state import CircuitQuantumStateMixin, QuantumState

if TYPE_CHECKING:
    import numpy.typing as npt

#: A type alias representing a numerical state vector,
#: equivalent to np.ndarray of complex floats.
StateVectorType: TypeAlias = "npt.NDArray[np.complex128]"


class QuantumStateVectorMixin(ABC):
    def __init__(
        self,
        n_qubits: int,
        vector: Optional[Union[StateVectorType, "npt.ArrayLike"]] = None,
    ) -> None:
        self._dim = 2**n_qubits
        self._vector: StateVectorType
        if vector is None:
            self._vector = cast(StateVectorType, np.zeros(self._dim))
            self._vector[0] = 1.0
        else:
            vector = np.asarray(vector, dtype=np.complex128)
            if len(vector) != self._dim:
                raise ValueError(f"The dimension of vector must be {self._dim}.")
            self._vector = vector

    @property
    def vector(self) -> StateVectorType:
        return readonly_array(self._vector)

    def draw(
        self,
        output: str = "text",
        *,
        max_terms: int = 16,
        threshold: float = 1e-10,
        precision: int = 6,
    ) -> Any:
        """Render the statevector.

        Args:
            output: One of ``"text"``, ``"latex_source"``, ``"latex"``,
                ``"qsphere"``, ``"density_matrix"``, or ``"bloch_vector"``. The ``"latex"`` option
                wraps the source in :class:`IPython.display.Latex` when IPython
                is available; otherwise the raw string is returned. The
                visualization options return a Matplotlib figure.
            max_terms: Maximum number of non-negligible terms to include. The rest
                are truncated.
            threshold: Terms with magnitude below this value are treated as 0.
            precision: Significant digits used for amplitudes/probabilities.
        """
        _warn_if_circuit_present(self)

        fmt = output.lower()
        if fmt not in {
            "text",
            "latex",
            "latex_source",
            "qsphere",
            "density_matrix",
            "bloch_vector",
        }:
            raise ValueError(f"Unsupported output format: {output}")

        terms = _significant_terms(self._vector, threshold)
        n_qubits = int(np.log2(self._dim))

        if fmt == "text":
            return _draw_text(terms, n_qubits, max_terms, precision)
        if fmt == "qsphere":
            return _draw_qsphere(self._vector, n_qubits, precision)
        if fmt == "density_matrix":
            return _draw_density_matrix(self._vector, n_qubits, precision)
        if fmt == "bloch_vector":
            return _draw_bloch_vector(self._vector, n_qubits)
        latex_src = _draw_latex(terms, n_qubits, max_terms, precision)
        if fmt == "latex_source":
            return latex_src
        return _to_ipython_latex(latex_src)

    def _repr_latex_(self) -> str:
        # Wrap in $$ so Jupyter renders math; keep it short with truncation defaults
        return f"$$ {self.draw(output='latex_source')} $$"


class QuantumStateVector(
    QuantumStateVectorMixin, CircuitQuantumStateMixin, QuantumState
):
    """QuantumStateVector represents a state defined by a state vector with an
    optional circuit to be applied."""

    def __init__(
        self,
        n_qubits: int,
        vector: Optional[Union[StateVectorType, "npt.ArrayLike"]] = None,
        circuit: Optional[ImmutableQuantumCircuit] = None,
    ):
        self._n_qubits = n_qubits
        QuantumStateVectorMixin.__init__(self, n_qubits, vector)
        CircuitQuantumStateMixin.__init__(self, n_qubits, circuit)

    def __repr__(self) -> str:
        return (
            f"QuantumStateVector(n_qubits={self.qubit_count}, vector={self.vector}, "
            f"circuit={self.circuit})"
        )

    @property
    def qubit_count(self) -> int:
        return self._n_qubits

    def with_gates_applied(self, gates: GateSequence) -> "QuantumStateVector":
        """Returns a new state with the gates applied.

        The original state is not changed.
        """
        circuit = self._circuit + gates
        return QuantumStateVector(self._n_qubits, self.vector, circuit)


def _warn_if_circuit_present(state: object) -> None:
    """Warn if draw() is called while a circuit with gates is attached.

    QuantumStateVector stores amplitudes directly; attached circuits are
    ignored during rendering to avoid accidentally suggesting that gates
    are applied.
    """
    circuit = getattr(state, "_circuit", None)
    gates = getattr(circuit, "gates", None)
    if gates and len(gates) > 0:
        warnings.warn(
            "draw() renders the stored state vector and ignores any attached circuit.",
            UserWarning,
            stacklevel=2,
        )


def _format_complex_text(val: complex, precision: int) -> str:
    real = val.real
    imag = val.imag
    real_zero = np.isclose(real, 0.0, atol=10 ** (-(precision + 2)))
    imag_zero = np.isclose(imag, 0.0, atol=10 ** (-(precision + 2)))

    parts: list[str] = []
    if not real_zero:
        parts.append(f"{real:.{precision}g}")
    if not imag_zero:
        sign = "+" if imag >= 0 and parts else ""
        parts.append(f"{sign}{imag:.{precision}g}j")
    if not parts:
        return "0"
    return "".join(parts)


def _format_complex_latex(val: complex, precision: int) -> str:
    real = val.real
    imag = val.imag
    real_zero = np.isclose(real, 0.0, atol=10 ** (-(precision + 2)))
    imag_zero = np.isclose(imag, 0.0, atol=10 ** (-(precision + 2)))

    parts: list[str] = []
    if not real_zero:
        parts.append(_format_component_latex(real, precision))
    if not imag_zero:
        sign = "+" if imag >= 0 and parts else ""
        imag_str = _format_component_latex(imag, precision)
        parts.append(f"{sign}{imag_str}i")
    if not parts:
        return "0"
    if len(parts) > 1:
        return f"({''.join(parts)})"
    return "".join(parts)


def _format_component_latex(val: float, precision: int) -> str:
    """Render a real component with a few common symbolic forms."""
    common = {
        (1 / np.sqrt(2)): r"\frac{1}{\sqrt{2}}",
        -(1 / np.sqrt(2)): r"-\frac{1}{\sqrt{2}}",
        np.pi / 2: r"\frac{\pi}{2}",
        -np.pi / 2: r"-\frac{\pi}{2}",
        np.pi / 4: r"\frac{\pi}{4}",
        -np.pi / 4: r"-\frac{\pi}{4}",
        np.pi / 6: r"\frac{\pi}{6}",
        -np.pi / 6: r"-\frac{\pi}{6}",
        np.pi / 8: r"\frac{\pi}{8}",
        -np.pi / 8: r"-\frac{\pi}{8}",
    }
    for k, v in common.items():
        if np.isclose(val, k, atol=10 ** (-(precision + 2))):
            return v
    return f"{val:.{precision}g}"


def _significant_terms(
    vector: StateVectorType, threshold: float
) -> list[tuple[int, complex]]:
    idxs = np.where(np.abs(vector) > threshold)[0]
    terms = [(int(idx), complex(vector[idx])) for idx in idxs]
    # Sort by magnitude descending, then by index for determinism
    terms.sort(key=lambda t: (-abs(t[1]), t[0]))
    return terms


def _draw_text(
    terms: list[tuple[int, complex]],
    n_qubits: int,
    max_terms: int,
    precision: int,
) -> str:
    if not terms:
        return "0"

    rendered = []
    for idx, amp in terms[:max_terms]:
        bitstr = format(idx, f"0{n_qubits}b")
        prob = abs(amp) ** 2
        rendered.append(
            f"|{bitstr}> amp={_format_complex_text(amp, precision)} "
            f"prob={prob:.{precision}g}"
        )
    if len(terms) > max_terms:
        truncated = len(terms) - max_terms
        rendered.append(f"... ({truncated} terms truncated)")
    return "\n".join(rendered)


def _draw_latex(
    terms: list[tuple[int, complex]],
    n_qubits: int,
    max_terms: int,
    precision: int,
) -> str:
    if not terms:
        return "0"

    rendered = []
    for idx, amp in terms[:max_terms]:
        bitstr = format(idx, f"0{n_qubits}b")
        rendered.append(f"{_format_complex_latex(amp, precision)}" f"|{bitstr}\\rangle")
    if len(terms) > max_terms:
        rendered.append(r"\cdots")
    return " + ".join(rendered)


def _to_ipython_latex(latex_str: str) -> Any:
    """Wrap LaTeX source for IPython display if available."""
    try:
        from IPython.display import Latex

        return Latex(f"$${latex_str}$$")
    except Exception:
        return latex_str


def _is_mpl_in_inline_mode() -> bool:
    from matplotlib import get_backend

    return get_backend() in ["nbAgg", "module://matplotlib_inline.backend_inline"]


def _draw_qsphere(
    vector: StateVectorType, n_qubits: int, precision: int
) -> Any:  # pragma: no cover - visual
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - import guard
        raise ImportError(
            "Matplotlib is required for 'qsphere' output. Install matplotlib to enable."
        ) from e
    try:
        import colorcet as cc  # type: ignore[import-untyped]

        cmap = cc.m_cyclic_mybm_20_100_c48
        # cmap = plt.get_cmap("hsv")
    except Exception:  # pragma: no cover - import guard
        cmap = plt.get_cmap("hsv")

    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(5, 4))
    ax = cast(Any, fig.add_subplot(111, projection="3d"))

    probs = np.abs(vector) ** 2
    phases = np.angle(vector)
    phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)

    # sphere grid
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.2, linewidth=0)

    top_indices = np.argsort(probs)[::-1]
    for i in top_indices:
        if probs[i] < 1e-12:
            continue
        bitstr = format(i, f"0{n_qubits}b")
        x, y, z = _qsphere_coordinates(bitstr, n_qubits)
        color = (phases[i] + 11 / 8 * np.pi) % (2 * np.pi) - np.pi
        size = max(20, 800 * probs[i])

        ax.scatter(
            [x],
            [y],
            [z],
            s=size,
            c=[color],
            linewidth=0.0,
            cmap=cmap,
            norm=phase_norm,
        )
        ax.scatter(
            [x],
            [y],
            [z],
            s=1.0,
            c="black",
        )

        ax.text(
            1.3 * x,
            1.3 * y,
            1.3 * z,
            f"$|{bitstr}\\rangle$",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

    # Add dashed latitude circles at Hamming-weight bands
    for weight in range(n_qubits + 1):
        z = -2 * weight / n_qubits + 1
        r = np.sqrt(max(0.0, 1 - z**2))
        theta = np.linspace(-2 * np.pi, 2 * np.pi, 200)
        lat_x = r * np.cos(theta)
        lat_y = r * np.sin(theta)
        lat_z = np.full_like(theta, z)
        ax.plot(lat_x, lat_y, lat_z, color="gray", lw=0.8, ls=":", alpha=0.5)

    # Phase color bar with pi/4 ticks
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=phase_norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05)
    ticks = [
        -np.pi,
        -3 * np.pi / 4,
        -np.pi / 2,
        -np.pi / 4,
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
        np.pi,
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(
        [
            r"$-\pi$",
            r"$-3\pi/4$",
            r"$-\pi/2$",
            r"$-\pi/4$",
            r"$0$",
            r"$\pi/4$",
            r"$\pi/2$",
            r"$3\pi/4$",
            r"$\pi$",
        ]
    )
    cbar.set_label("Phase", rotation=270, labelpad=15)

    # Keep perspective level with the equator and aspect spherical
    ax.view_init(elev=5, azim=90)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_axis_off()
    fig.tight_layout()

    if _is_mpl_in_inline_mode():
        plt.close(fig)
    return fig


def _draw_density_matrix(
    vector: StateVectorType, n_qubits: int, precision: int
) -> Any:  # pragma: no cover - visual
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as e:  # pragma: no cover - import guard
        raise ImportError(
            "Matplotlib is required for 'density_matrix' output. Install matplotlib to enable."
        ) from e

    rho = np.outer(vector, np.conjugate(vector))
    dim = rho.shape[0]
    max_abs = np.max(np.abs(rho)) or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = [format(i, f"0{n_qubits}b") for i in range(dim)]

    def _hinton(ax: Any, data: NDArray[Any], title: str) -> None:
        ax.patch.set_facecolor("gray")
        ax.set_aspect("equal", "box")
        ax.set_xticks(np.arange(dim) + 0.5)
        ax.set_yticks(np.arange(dim) + 0.5)
        ax.set_xticklabels([f"$|{label}\\rangle$" for label in labels])
        ax.set_yticklabels([f"$|{label}\\rangle$" for label in labels])
        ax.set_xlim(0, dim)
        ax.set_ylim(0, dim)
        ax.set_title(title)
        for (r, c), val in np.ndenumerate(data[::-1, :]):
            magnitude = np.sqrt(abs(val) / max_abs)
            if magnitude == 0:
                continue
            plot_x, plot_y = c + 0.5, r + 0.5
            color = "white" if val.real >= 0 else "black"
            rect = Rectangle(
                (plot_x - magnitude / 2, plot_y - magnitude / 2),
                magnitude,
                magnitude,
                facecolor=color,
                edgecolor=color,
            )
            ax.add_patch(rect)

    _hinton(axes[0], rho.real, "Re[ρ]")
    _hinton(axes[1], rho.imag, "Im[ρ]")

    fig.tight_layout()
    if _is_mpl_in_inline_mode():
        plt.close(fig)
    return fig


def _draw_bloch_vector(
    vector: StateVectorType, n_qubits: int
) -> Any:  # pragma: no cover - visual
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - import guard
        raise ImportError("Matplotlib is required for 'bloch_vector' output.") from e

    bloch_vecs = _bloch_vectors_from_state(vector, n_qubits)
    cols = min(4, n_qubits)
    rows = int(np.ceil(n_qubits / cols))
    fig, axes = plt.subplots(
        rows, cols, subplot_kw={"projection": "3d"}, figsize=(cols * 4, rows * 4)
    )
    axes_list = cast(Any, np.atleast_1d(axes).flatten())

    for idx, bloch_vec in enumerate(bloch_vecs):
        ax = cast(Any, axes_list[idx])
        ax.view_init(elev=25, azim=30)
        _render_bloch_vector(ax, bloch_vec)
        ax.set_title(f"Qubit {idx}", fontsize=10)

        # Equator and longitude markers
        theta = np.linspace(0, 2 * np.pi, 400)
        for phi in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            eq_x = np.cos(theta) * np.sin(phi)
            eq_y = np.sin(theta) * np.sin(phi)
            eq_z = np.ones_like(theta) * np.cos(phi)
            ax.plot(eq_x, eq_y, eq_z, color="gray", lw=0.8, ls=":", alpha=0.7)

        for phi in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            lon_x = np.cos(phi) * np.sin(theta)
            lon_y = np.sin(phi) * np.sin(theta)
            lon_z = np.cos(theta)
            ax.plot(lon_x, lon_y, lon_z, color="gray", lw=0.8, ls=":", alpha=0.7)

    # Hide unused subplots
    for ax in axes_list[len(bloch_vecs) :]:
        ax.set_visible(False)

    fig.tight_layout()
    if _is_mpl_in_inline_mode():
        plt.close(fig)
    return fig


def _qsphere_coordinates(bitstr: str, n_qubits: int) -> tuple[float, float, float]:
    """Position basis states on a sphere."""
    weight = bitstr.count("1")
    z = -2 * weight / n_qubits + 1
    num_divisions = comb(n_qubits, weight) if weight else 1
    order = _bit_string_index(bitstr)
    angle = (weight / n_qubits) * (2 * np.pi) + (order * 2 * (np.pi / num_divisions))
    if (weight > n_qubits / 2) or (
        weight == n_qubits / 2 and order >= num_divisions / 2
    ):
        angle = np.pi - angle - (2 * np.pi / num_divisions)

    r = np.sqrt(max(0.0, 1 - z**2))
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return x, y, z


def _bit_string_index(bitstr: str) -> int:
    """Lexicographic index among strings with the same Hamming weight."""
    n = len(bitstr)
    ones = [pos for pos, char in enumerate(bitstr) if char == "1"]
    k = len(ones)
    dualm = sum(comb(n - 1 - ones[k - 1 - i], i + 1) for i in range(k))
    return int(dualm)


def _bloch_vectors_from_state(
    vector: StateVectorType, n_qubits: int
) -> list[tuple[float, float, float]]:
    """Compute single-qubit Bloch vectors for each qubit of a pure
    statevector."""
    vec = np.asarray(vector, dtype=np.complex128)
    dim = len(vec)
    if dim != 2**n_qubits:
        raise ValueError("Statevector dimension does not match qubit count.")

    bloch_vecs: list[tuple[float, float, float]] = []
    for q in range(n_qubits):
        sx = 0.0
        sy = 0.0
        sz = 0.0
        mask = 1 << q
        for idx in range(dim):
            amp = vec[idx]
            if idx & mask:
                sz -= abs(amp) ** 2
            else:
                sz += abs(amp) ** 2
                partner = idx | mask
                partner_amp = vec[partner]
                prod = np.conjugate(amp) * partner_amp
                sx += 2 * np.real(prod)
                sy += 2 * np.imag(prod)
        bloch_vecs.append((sx, sy, sz))
    return bloch_vecs


def _render_bloch_vector(ax: Any, bloch_vec: tuple[float, float, float]) -> None:
    """Render a single Bloch vector on a unit sphere."""
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d  # type: ignore[import-untyped]

    # Draw sphere surface
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color="lightgray", alpha=0.2, linewidth=0
    )

    # Axes lines
    ax.plot([-1, 1], [0, 0], [0, 0], color="grey", linewidth=1)
    ax.plot([0, 0], [-1, 1], [0, 0], color="grey", linewidth=1)
    ax.plot([0, 0], [0, 0], [-1, 1], color="grey", linewidth=1)
    ax.text(1.3, 0, 0, "X", color="grey")
    ax.text(0, 1.1, 0, "Y", color="grey")
    ax.text(0, 0, 1.1, "Z", color="grey")

    class Arrow3D(FancyArrowPatch):
        def __init__(
            self, xs: Any, ys: Any, zs: Any, *args: Any, **kwargs: Any
        ) -> None:
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer: Any = None) -> float:
            xs3d, ys3d, zs3d = self._verts3d
            axes = cast(Any, self.axes)
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return float(np.min(zs))

    sx, sy, sz = bloch_vec
    arrow = Arrow3D(
        [0, sx],
        [0, sy],
        [0, sz],
        mutation_scale=20,
        lw=5,
        arrowstyle="-|>",
        color="darkblue",
    )
    ax.add_artist(arrow)

    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
