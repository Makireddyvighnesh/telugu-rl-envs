"""Plots for the README, from results/test_results.csv and results/training_curves.csv.

    python scripts/plot_results.py            # writes results/plots/<name>-{light,dark}.png

Each figure is rendered twice so GitHub can show the one matching the reader's theme.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "plots"

# Categorical slots 1-4 (blue, orange, aqua, yellow), validated for CVD separation on both surfaces.
THEMES = {
    "light": dict(bg="#fcfcfb", fg="#0b0b0b", fg2="#52514e", grid="#e7e6e2", muted="#bcbab3",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(bg="#1a1a19", fg="#f0efec", fg2="#c3c2b7", grid="#34332f", muted="#5d5c57",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}
# Color follows the run; the 200-step runs are also dashed so identity never rests on color alone.
RUNS = [
    ("GRPO 1.5B, 1000 steps", "-"),
    ("OPD 0.5B, 1000 steps", "-"),
    ("GRPO 0.5B, 200 steps", "--"),
    ("OPD 0.5B, 200 steps", "--"),
]
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


def style(ax, t):
    ax.set_facecolor(t["bg"])
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["fg2"], labelsize=8.5, length=0)


def rolling(vals, w):
    out, s = [], 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= w:
            s -= vals[i - w]
        out.append(s / min(i + 1, w))
    return out


def load_curves():
    by_run = defaultdict(list)
    with open(RESULTS / "training_curves.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_run[r["run"]].append(r)
    return by_run


def series(rows, key, w, max_step=None):
    pts = [(int(r["step"]), float(r[key])) for r in rows if r[key] != "" and (max_step is None or int(r["step"]) <= max_step)]
    pts.sort()
    return [p[0] for p in pts], rolling([p[1] for p in pts], w)


def test_results(t):
    with open(RESULTS / "test_results.csv", encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: float(r["exact_match"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.9), facecolor=t["bg"])
    style(ax, t)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    labels = [f"{r['model']} · {r['training']}" for r in rows]
    vals = [float(r["exact_match"]) for r in rows]
    colors = [t["muted"] if r["training"] == "base" else t["series"][0] for r in rows]
    ax.barh(range(len(rows)), vals, height=0.68, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.1%}", va="center", fontsize=8.5, color=t["fg"])
    ax.set_yticks(range(len(rows)), labels, color=t["fg"], fontsize=8.5)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel("Exact match, greedy, 10,260 test words", color=t["fg2"], fontsize=8.5)
    leg = fig.legend([Patch(color=t["series"][0]), Patch(color=t["muted"])], ["Trained", "Base model"],
                     loc="upper left", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.005, 1.0))
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def training_curves(t, by_run):
    panels = [("reward", "Train reward", None), ("exact", "Exact match", PCT),
              ("entropy", "Entropy (nats)", None), ("zero_signal", "Zero-signal groups (GRPO only)", PCT)]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), facecolor=t["bg"], sharex=True)
    for ax, (key, title, fmt) in zip(axes.flat, panels):
        style(ax, t)
        for i, (run, ls) in enumerate(RUNS):
            x, y = series(by_run[run], key, 20)
            if x:
                ax.plot(x, y, color=t["series"][i], linewidth=1.6, linestyle=ls, solid_capstyle="round")
        ax.set_title(title, loc="left", color=t["fg"], fontsize=10, pad=6)
        if fmt:
            ax.yaxis.set_major_formatter(fmt)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 1000)
    for ax in axes[1]:
        ax.set_xlabel("Training step", color=t["fg2"], fontsize=8.5)
    handles = [Line2D([], [], color=t["series"][i], linewidth=1.8, linestyle=ls) for i, (_, ls) in enumerate(RUNS)]
    leg = fig.legend(handles, [r for r, _ in RUNS], loc="upper center", ncol=4, frameon=False, fontsize=8.5,
                     handlelength=2.4, bbox_to_anchor=(0.5, 1.0))
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    fig.text(0.99, 0.005, "20-step rolling mean", color=t["fg2"], fontsize=7.5, ha="right")
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    return fig


def first_200(t, by_run):
    runs = [("GRPO 0.5B, 200 steps", 2), ("OPD 0.5B, 200 steps", 3)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9), facecolor=t["bg"])
    for ax, (key, title, fmt) in zip(axes, [("reward", "Train reward", None), ("exact", "Exact match", PCT)]):
        style(ax, t)
        ends = []
        for run, slot in runs:
            x, y = series(by_run[run], key, 5)
            ax.plot(x, y, color=t["series"][slot], linewidth=1.6, solid_capstyle="round")
            ax.plot(x[-1], y[-1], "o", color=t["series"][slot], markersize=4)
            ends.append([y[-1], run.split(",")[0].replace(" 0.5B", ""), slot])
        ends.sort()
        if len(ends) == 2 and ax.get_ylim()[1] and abs(ends[1][0] - ends[0][0]) < 0.06 * ax.get_ylim()[1]:
            ends[1][0] = ends[0][0] + 0.06 * ax.get_ylim()[1]  # keep the two end labels from meeting
        for yv, name, slot in ends:
            label = f"{name}  {yv:.0%}" if fmt else f"{name}  {yv:.2f}"
            ax.text(208, yv, label, va="center", fontsize=8.5, color=t["fg"])
        ax.set_title(title, loc="left", color=t["fg"], fontsize=10, pad=6)
        if fmt:
            ax.yaxis.set_major_formatter(fmt)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 255)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_xlabel("Training step", color=t["fg2"], fontsize=8.5)
    fig.text(0.99, 0.005, "Same 0.5B base, 64 completions/step · 5-step rolling mean", color=t["fg2"],
             fontsize=7.5, ha="right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_run = load_curves()
    for mode, t in THEMES.items():
        for name, fig in (("test-results", test_results(t)), ("training-curves", training_curves(t, by_run)),
                          ("grpo-vs-opd-05b", first_200(t, by_run))):
            path = OUT / f"{name}-{mode}.png"
            fig.savefig(path, dpi=200, facecolor=t["bg"])
            plt.close(fig)
            print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
