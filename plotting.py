"""Tema grafico condiviso per le figure matplotlib/seaborn (chiaro/scuro)."""
from __future__ import annotations

import io
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")  # le figure vengono incorporate in Tk tramite FigureCanvasTkAgg
import matplotlib as mpl  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

PALETTE = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6",
           "#14b8a6", "#ec4899", "#84cc16", "#f97316", "#6366f1"]
ACCENT = PALETTE[0]
FORECAST = PALETTE[1]


def theme_rc(dark: bool) -> dict:
    bg = "#242424" if dark else "#ffffff"
    fg = "#e6e6e6" if dark else "#1f2328"
    grid = "#3a3a3a" if dark else "#e5e7eb"
    return {
        "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
        "axes.edgecolor": grid, "axes.labelcolor": fg, "axes.titlecolor": fg,
        "xtick.color": fg, "ytick.color": fg, "text.color": fg,
        "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "legend.frameon": False, "font.size": 9.5, "axes.titlesize": 11,
        "axes.titleweight": "bold", "figure.autolayout": False,
    }


@contextmanager
def themed(dark: bool):
    with mpl.rc_context(theme_rc(dark)):
        yield


def new_figure(dark: bool, size=(8, 4.8)) -> Figure:
    """Da chiamare dentro `themed(dark)`."""
    return Figure(figsize=size, dpi=100, layout="constrained")


def figure_png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    return buf.getvalue()


def short(label, n: int = 22) -> str:
    text = str(label)
    return text if len(text) <= n else text[: n - 1] + "…"
