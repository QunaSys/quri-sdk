# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
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


_IDENTIFICATION_BASE_CONSTANTS = ["pi", "sqrt(2)", "sqrt(3)", "sqrt(5)"]
_LATEX_NAME_MAP = {"pi": r"\pi", "e": "e"}


def _format_component_latex(val: float, precision: int) -> str:
    """Render a real component using mpmath-based identifications when
    possible."""
    identified = _identify_component_latex(val, precision)
    if identified is not None:
        return identified
    return f"{val:.{precision}g}"


def _identify_component_latex(val: float, precision: int) -> Optional[str]:
    """Try to express the value symbolically and convert to LaTeX."""
    try:
        from mpmath import mp  # type: ignore[import-untyped]
    except Exception:
        return None

    tol = 10 ** (-(precision + 2))
    prev_dps = mp.dps
    try:
        mp.dps = max(prev_dps, precision + 5, 30)
        expr = mp.identify(mp.mpf(val), _IDENTIFICATION_BASE_CONSTANTS, tol=tol)
    except Exception:
        return None
    finally:
        mp.dps = prev_dps
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return None

    try:
        evaluated = _evaluate_identification_ast(tree.body, mp)
        target = mp.mpf(val)
        tol_val = mp.mpf(tol)
        if not mp.almosteq(evaluated, target, rel_eps=tol_val, abs_eps=tol_val):
            return None
    except Exception:
        return None

    return _identification_expr_to_latex(tree.body, precision)


def _identification_expr_to_latex(expr: ast.AST, precision: int) -> Optional[str]:
    """Convert the expression returned by mpmath.identify into LaTeX."""
    try:
        latex, _ = _ast_to_latex(expr, precision)
    except Exception:
        return None
    return latex


def _evaluate_identification_ast(node: ast.AST, mp: Any) -> Any:
    """Evaluate the identification AST using the given mpmath context."""
    if isinstance(node, ast.Constant):
        return mp.mpf(node.value)
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return mp.pi
        if node.id == "e":
            return mp.e
        raise ValueError(f"Unsupported constant name: {node.id}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate_identification_ast(node.operand, mp)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            return _evaluate_identification_ast(
                node.left, mp
            ) + _evaluate_identification_ast(node.right, mp)
        if isinstance(node.op, ast.Sub):
            return _evaluate_identification_ast(
                node.left, mp
            ) - _evaluate_identification_ast(node.right, mp)
        if isinstance(node.op, ast.Mult):
            return _evaluate_identification_ast(
                node.left, mp
            ) * _evaluate_identification_ast(node.right, mp)
        if isinstance(node.op, ast.Div):
            return _evaluate_identification_ast(
                node.left, mp
            ) / _evaluate_identification_ast(node.right, mp)
        if isinstance(node.op, ast.Pow):
            return _evaluate_identification_ast(
                node.left, mp
            ) ** _evaluate_identification_ast(node.right, mp)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_evaluate_identification_ast(arg, mp) for arg in node.args]
        func = node.func.id
        if func == "sqrt" and args:
            return mp.sqrt(args[0])
        if func == "log" and args:
            return mp.log(args[0])
        if func == "exp" and args:
            return mp.e ** args[0]
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def _ast_to_latex(node: ast.AST, precision: int) -> tuple[str, int]:
    """Recursively convert a restricted AST to a LaTeX string."""
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, float):
            return f"{val:.{precision}g}", 5
        return str(val), 5
    if isinstance(node, ast.Name):
        return _LATEX_NAME_MAP.get(node.id, node.id), 5
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner, inner_prec = _ast_to_latex(node.operand, precision)
        return f"-{_parenthesize(inner, inner_prec, 2)}", 4
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            left, left_prec = _ast_to_latex(node.left, precision)
            right, right_prec = _ast_to_latex(node.right, precision)
            return (
                f"{_parenthesize(left, left_prec, 1)} + {_parenthesize(right, right_prec, 1)}",
                1,
            )
        if isinstance(node.op, ast.Sub):
            left, left_prec = _ast_to_latex(node.left, precision)
            right, right_prec = _ast_to_latex(node.right, precision)
            return (
                f"{_parenthesize(left, left_prec, 1)} - {_parenthesize(right, right_prec, 1)}",
                1,
            )
        if isinstance(node.op, ast.Mult):
            left, left_prec = _ast_to_latex(node.left, precision)
            right, right_prec = _ast_to_latex(node.right, precision)
            left = _parenthesize(left, left_prec, 2)
            right = _parenthesize(right, right_prec, 2)
            return _combine_multiplication_terms(left, right), 2
        if isinstance(node.op, ast.Div):
            numerator, _ = _ast_to_latex(node.left, precision)
            denominator, _ = _ast_to_latex(node.right, precision)
            return f"\\frac{{{numerator}}}{{{denominator}}}", 3
        if isinstance(node.op, ast.Pow):
            base, base_prec = _ast_to_latex(node.left, precision)
            exp, exp_prec = _ast_to_latex(node.right, precision)
            base = _parenthesize(base, base_prec, 3)
            exp = _parenthesize(exp, exp_prec, 3)
            return f"{base}^{{{exp}}}", 3
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_ast_to_latex(arg, precision) for arg in node.args]
        func = node.func.id
        if func == "sqrt" and args:
            arg, _ = args[0]
            return f"\\sqrt{{{arg}}}", 4
        if func == "log" and args:
            arg, _ = args[0]
            return f"\\log\\left({arg}\\right)", 4
        if func == "exp" and args:
            arg, _ = args[0]
            return f"e^{{{arg}}}", 4
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def _parenthesize(expr: str, child_prec: int, parent_prec: int) -> str:
    if child_prec < parent_prec:
        return f"({expr})"
    return expr


def _combine_multiplication_terms(left: str, right: str) -> str:
    denom = _unit_fraction_denominator(left)
    if denom is not None:
        return f"\\frac{{{right}}}{{{denom}}}"
    denom = _unit_fraction_denominator(right)
    if denom is not None:
        return f"\\frac{{{left}}}{{{denom}}}"
    if left == "1":
        return right
    if left == "-1":
        return f"-{right}"
    return f"{left}\\,{right}"


def _unit_fraction_denominator(expr: str) -> Optional[str]:
    if expr.startswith("\\frac{1}{") and expr.endswith("}"):
        return expr[len("\\frac{1}{") : -1]
    return None


def _significant_terms(
    vector: StateVectorType, threshold: float
) -> list[tuple[int, complex]]:
    idxs = np.where(np.abs(vector) > threshold)[0]
    terms = [(int(idx), complex(vector[idx])) for idx in idxs]
    # # Sort by magnitude descending, then by index for determinism
    # terms.sort(key=lambda t: (-abs(t[1]), t[0]))
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
        import importlib

        cc = importlib.import_module("colorcet")
        cmap = cc.m_cyclic_mybm_20_100_c48
    except Exception:  # pragma: no cover - import guard
        cmap = plt.get_cmap("hsv")

    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(6, 4))
    ax = cast(Any, fig.add_subplot(111, projection="3d", computed_zorder=True))

    probs = np.abs(vector) ** 2
    phases = np.angle(vector)
    phase_norm = Normalize(vmin=0, vmax=2 * np.pi)

    # sphere grid
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.2, linewidth=0, zorder=0)

    top_indices = np.argsort(probs)[::-1]
    for i in top_indices:
        if probs[i] < 1e-12:
            continue
        bitstr = format(i, f"0{n_qubits}b")
        x, y, z = _qsphere_coordinates(bitstr, n_qubits)
        color = (phases[i] + 2 * np.pi) % (2 * np.pi)
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

        ax.text(
            1.3 * x,
            1.3 * y,
            1.3 * z,
            f"$|{bitstr}\\rangle$",
            ha="center",
            va="center",
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
        0,
        np.pi / 2,
        np.pi,
        3 * np.pi / 2,
        2 * np.pi,
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(
        [
            r"$0$",
            r"$\pi/2$",
            r"$\pi$",
            r"$3\pi/2$",
            r"$2\pi$",
        ]
    )
    cbar.set_label("Phase", rotation=270, labelpad=15)

    ax.view_init(elev=7, azim=80)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])

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
        for (r, c), val in np.ndenumerate(data[:, :]):
            magnitude = np.sqrt(abs(val))
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

        # Axes lines
        ax.plot([-1, 1], [0, 0], [0, 0], color="grey", linewidth=1)
        ax.plot([0, 0], [-1, 1], [0, 0], color="grey", linewidth=1)
        ax.plot([0, 0], [0, 0], [-1, 1], color="grey", linewidth=1)
        ax.text(1.1, 0.0, 0.1, "X", color="grey")
        ax.text(0.0, 1.0, 0.1, "Y", color="grey")
        ax.text(0.0, -0.05, 1.05, "Z", color="grey")

        # Equator and 90-degree meridians as reference
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(
            np.cos(theta),
            np.sin(theta),
            np.zeros_like(theta),
            color="gray",
            lw=0.8,
            ls=":",
            alpha=0.5,
        )
        t = np.linspace(0, np.pi, 400)
        for phi in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            ax.plot(
                np.cos(phi) * np.sin(t),
                np.sin(phi) * np.sin(t),
                np.cos(t),
                color="gray",
                lw=0.8,
                ls=":",
                alpha=0.7,
            )

        # Latitude circle and meridian at the Bloch vector position
        sx, sy, sz = bloch_vec
        magnitude = np.sqrt(sx**2 + sy**2 + sz**2)
        if magnitude > 1e-6:
            phi = np.arctan2(sy, sx)
            polar_angle = np.arccos(np.clip(sz / magnitude, -1.0, 1.0))
            theta = np.linspace(0, 2 * np.pi, 400)
            ax.plot(
                np.cos(theta) * np.sin(polar_angle),
                np.sin(theta) * np.sin(polar_angle),
                np.full_like(theta, np.cos(polar_angle)),
                color="orange",
                lw=0.8,
                ls="-",
                alpha=0.6,
            )
            
            t = np.linspace(0., np.pi, 200)
            ax.plot(
                np.cos(phi) * np.sin(t),
                np.sin(phi) * np.sin(t),
                np.cos(t),
                color="orange",
                lw=0.8,
                ls="-",
                alpha=0.6,
            )

            t = np.linspace(0., 1., 200)
            ax.plot(
                t * sx / magnitude,
                t * sy / magnitude,
                t * sz / magnitude,
                color="orange",
                lw=0.8,
                ls="--",
                alpha=0.6,
            )

            ax.scatter(
                np.cos(phi) * np.sin(polar_angle),
                np.sin(phi) * np.sin(polar_angle),
                np.cos(polar_angle),
                color="orange",
                lw=0.8,
                ls="-",
                alpha=0.6,
            )

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


def _bloch_arrow_surface(
    ax: Any,
    direction: tuple[float, float, float],
    width: float = 0.03,
    head_fraction: float = 0.2,
    head_width: float = 2.5,
    color: str = "orange",
) -> None:
    """Draw a 3D arrow as a surface of revolution pointing in *direction*.

    The arrow tip lands exactly at *direction* (which doubles as the endpoint),
    and the surface participates in matplotlib's z-ordering so it is correctly
    occluded by the sphere when pointing away from the viewer.
    """
    sx, sy, sz = direction
    length = float(np.sqrt(sx**2 + sy**2 + sz**2))
    if length < 1e-10:
        return

    # Head scales sublinearly with arrow length (exponent < 1) so it stays
    # relatively larger for short arrows.  Capped so it never exceeds the
    # total arrow length.
    nominal_head_length = head_fraction  # absolute size at length == 1
    nominal_head_radius = head_width * width
    scale = (length+2.) / 3.  # sublinear: shrinks slower than the arrow
    actual_head_length = min(length, nominal_head_length * scale)
    actual_head_radius = nominal_head_radius * scale
    shaft_length = length - actual_head_length
    shaft_width = scale * width

    # 2-D profile (radius, height) defining shaft + cone head
    profile_r = np.array([0, shaft_width, shaft_width, actual_head_radius, 0])
    profile_z = np.array([0, 0, shaft_length, shaft_length, length])

    theta = np.linspace(0, 2 * np.pi, 30)
    r_grid, theta_grid = np.meshgrid(profile_r, theta)
    z_grid = np.tile(profile_z, (len(theta), 1))
    x_grid = r_grid * np.sin(theta_grid)
    y_grid = r_grid * np.cos(theta_grid)

    # Rotate the z-aligned arrow to point along (sx, sy, sz)
    polar = float(np.arccos(np.clip(sz / length, -1.0, 1.0)))
    azimuth = float(np.arctan2(sx, -sy))
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(polar), -np.sin(polar)],
            [0, np.sin(polar), np.cos(polar)],
        ]
    )
    rot_z = np.array(
        [
            [np.cos(azimuth), -np.sin(azimuth), 0],
            [np.sin(azimuth), np.cos(azimuth), 0],
            [0, 0, 1],
        ]
    )
    pts = np.c_[x_grid.flatten(), y_grid.flatten(), z_grid.flatten()].T
    rotated = (rot_z @ rot_x @ pts).T
    from matplotlib.colors import LightSource

    ax.plot_surface(
        rotated[:, 0].reshape(r_grid.shape),
        rotated[:, 1].reshape(r_grid.shape),
        rotated[:, 2].reshape(r_grid.shape),
        color=color,
        alpha=0.9,
        linewidth=0,
        shade=True,
        lightsource=LightSource(azdeg=45, altdeg=-45),
    )


def _render_bloch_vector(ax: Any, bloch_vec: tuple[float, float, float]) -> None:
    """Render a single Bloch vector on a unit sphere."""
    # Draw sphere surface
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color="lightgray", alpha=0.2, linewidth=0
    )

    _bloch_arrow_surface(ax, bloch_vec)

    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
