"""Shared HSLU event palette for the Streamlit app and project notebook."""
import matplotlib as mpl

NOTEBOOK_COLORS = {
    "primary": "#0077C8",
    "secondary": "#66B5E9",
    "reference": "#D0384B",
    "dark": "#153652",
    "teal": "#16856F",
    "orange": "#C88126",
    "purple": "#6861B4",
    "gray": "#718196",
    "neutral": "#192B3A",
}
NOTEBOOK_PALETTE = [
    NOTEBOOK_COLORS[key]
    for key in ("primary", "secondary", "teal", "orange", "purple", "gray", "dark", "neutral")
]


def apply_plot_style() -> None:
    """Change presentation only; no data, estimators, or randomness are touched."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.figsize": (8, 4.8),
        "figure.dpi": 110,
        "figure.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#C8DCEC",
        "axes.labelcolor": "#4C6280",
        "axes.titlecolor": "#111A28",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 16,
        "axes.labelsize": 10,
        "axes.labelpad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=NOTEBOOK_PALETTE),
        "grid.color": "#D6E7F5",
        "grid.alpha": .65,
        "grid.linewidth": .6,
        "xtick.color": "#4C6280",
        "ytick.color": "#4C6280",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "text.color": "#111A28",
        "lines.linewidth": 2,
        "patch.edgecolor": "#FFFFFF",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
