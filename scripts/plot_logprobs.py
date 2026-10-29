"""Plots and summary numbers for the token-level GRPO vs OPD analysis (scripts/analyze_logprobs.py output).

    python scripts/plot_logprobs.py --data outputs/logprob-analysis

Writes results/plots/logprobs-<name>-{light,dark}.png and results/logprob_summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from plot_results import THEMES, style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "plots"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")
LABEL = {"base": "Base 0.5B", "grpo-200": "GRPO 0.5B, 200 steps", "opd-200": "OPD 0.5B, 200 steps",
         "opd-1000": "OPD 0.5B, 1000 steps", "teacher": "Teacher (GRPO 1.5B)"}
MAIN = ["base", "grpo-200", "opd-200", "opd-1000", "teacher"]
CLASSES = ["consonant", "vowel sign", "virama", "nasal/visarga sign", "vowel", "end"]
CLASS_LABEL = {"consonant": "Consonant", "vowel sign": "Vowel sign", "virama": "Virama (conjunct)",
               "nasal/visarga sign": "Anusvara / visarga", "vowel": "Independent vowel", "end": "Stop (end of word)"}


def colors(t):
    # Method follows color everywhere: GRPO orange, OPD blue (aqua for the longer run), base muted, teacher ink.
    return {"base": t["muted"], "grpo-200": t["series"][1], "opd-200": t["series"][0],
            "opd-1000": t["series"][2], "teacher": t["fg"],
            "grpo-100": t["series"][1], "opd-100": t["series"][0]}


def legend(fig, t, names, y=1.0, ncol=None):
    c = colors(t)
    handles = [Line2D([], [], color=c[n], linewidth=2, linestyle="--" if n == "teacher" else "-", marker="o",
                      markersize=4) for n in names]
    leg = fig.legend(handles, [LABEL[n] for n in names], loc="upper center", ncol=ncol or len(names), frameon=False,
                     fontsize=8.5, bbox_to_anchor=(0.5, y))
    for txt in leg.get_texts():
        txt.set_color(t["fg"])


def char_table(ref):
    """Per (model, word, character): log-prob of the correct character = sum of its tokens' log-probs."""
    return ref.groupby(["model", "word_idx", "char_idx", "char_class"], as_index=False).agg(
        logp=("logp", "sum"), t_logp=("t_logp", "sum"), kl=("kl", "mean"), entropy=("entropy", "mean"),
        t_entropy=("t_entropy", "mean"))


def fig_word_prob(t, words):
    c = colors(t)
    fig, ax = plt.subplots(figsize=(7.6, 3.6), facecolor=t["bg"])
    style(ax, t)
    for n in MAIN:
        p = np.sort(np.exp(words.loc[words.model == n, "ref_logp"].to_numpy()))[::-1]
        share = np.arange(1, len(p) + 1) / len(p)
        ax.plot(share, p, color=c[n], linewidth=1.8, linestyle="--" if n == "teacher" else "-")
    ax.set_xlabel("Share of test words", color=t["fg2"], fontsize=8.5)
    ax.set_ylabel("P(correct word)", color=t["fg2"], fontsize=8.5)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 0.3)
    ax.set_title("Probability each model gives the correct answer (words sorted, most confident first)",
                 loc="left", color=t["fg"], fontsize=10, pad=6)
    ax.text(0.297, 0.97, "remaining words: P ≈ 0 for every model", color=t["fg2"], fontsize=7.5, ha="right", va="top")
    legend(fig, t, MAIN, y=1.0, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def fig_delta(t, chars):
    """Per correct character: change in log-prob vs base, GRPO-200 on x, OPD-200 on y."""
    piv = chars.pivot_table(index=["word_idx", "char_idx"], columns="model", values="logp")
    dg = (piv["grpo-200"] - piv["base"]).clip(-8, 12)
    do = (piv["opd-200"] - piv["base"]).clip(-8, 12)
    fig, ax = plt.subplots(figsize=(5.4, 5.0), facecolor=t["bg"])
    style(ax, t)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    hb = ax.hexbin(dg, do, gridsize=45, bins="log", mincnt=1, cmap=mcolors.LinearSegmentedColormap.from_list(
        "m", [t["grid"], t["series"][0]]), linewidths=0)
    lim = (-8, 12)
    ax.plot(lim, lim, color=t["fg2"], linewidth=1, linestyle="--")
    ax.axhline(0, color=t["fg2"], linewidth=0.6)
    ax.axvline(0, color=t["fg2"], linewidth=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    above = float((do > dg).mean())
    ax.text(-7.5, 11.2, f"OPD gained more: {above:.0%} of letters", color=t["fg"], fontsize=8.5, va="top")
    ax.text(11.5, -7.5, f"GRPO gained more: {1 - above:.0%}", color=t["fg"], fontsize=8.5, ha="right")
    ax.set_xlabel("GRPO: change in log P(correct letter) vs base", color=t["fg2"], fontsize=8.5)
    ax.set_ylabel("OPD: change in log P(correct letter) vs base", color=t["fg2"], fontsize=8.5)
    ax.set_title("How each method moved every correct letter (200 steps each)", loc="left", color=t["fg"],
                 fontsize=10, pad=6)
    cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors=t["fg2"], labelsize=7.5)
    cb.outline.set_visible(False)
    cb.set_label("letters (log scale)", color=t["fg2"], fontsize=7.5)
    fig.tight_layout()
    return fig


def fig_by_class(t, chars):
    c = colors(t)
    names = ["base", "grpo-200", "opd-200", "teacher"]
    m = chars.assign(p=np.exp(chars.logp)).groupby(["char_class", "model"]).p.mean().unstack()
    classes = [k for k in CLASSES if k in m.index]
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(classes) + 1.4), facecolor=t["bg"])
    style(ax, t)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    for i, k in enumerate(classes):
        ax.plot([m.loc[k, names].min(), m.loc[k, names].max()], [i, i], color=t["grid"], linewidth=3, zorder=1)
        for n in names:
            ax.plot(m.loc[k, n], i, "o", color=c[n], markersize=7 if n != "teacher" else 6, zorder=3,
                    markerfacecolor=c[n] if n != "teacher" else t["bg"], markeredgewidth=1.6)
    counts = chars[chars.model == "base"].char_class.value_counts()
    ax.set_yticks(range(len(classes)), [f"{CLASS_LABEL[k]}  ({counts.get(k, 0):,})" for k in classes],
                  color=t["fg"], fontsize=8.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean P(correct letter), teacher-forced on the reference", color=t["fg2"], fontsize=8.5)
    legend(fig, t, names, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def fig_alignment(t, greedy):
    c = colors(t)
    g = greedy.assign(rejected=np.exp(greedy.t_logp) < 0.1)
    per = g.groupby("model").agg(kl=("kl", "mean"), agree=("agree", "mean"), rejected=("rejected", "mean"))
    runs = {"GRPO": ["base", "grpo-100", "grpo-200"], "OPD": ["base", "opd-100", "opd-200", "opd-1000"]}
    steps = {"base": 0, "grpo-100": 100, "grpo-200": 200, "opd-100": 100, "opd-200": 200, "opd-1000": 1000}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), facecolor=t["bg"])
    panels = [("kl", "Reverse KL to teacher (nats/token)", None),
              ("agree", "Teacher's top token = model's", PCT),
              ("rejected", "Chosen tokens the teacher gives < 10%", PCT)]
    for ax, (key, title, fmt) in zip(axes, panels):
        style(ax, t)
        for method, names in runs.items():
            names = [n for n in names if n in per.index]
            col = c["grpo-200"] if method == "GRPO" else c["opd-200"]
            xs, ys = [steps[n] for n in names], [per.loc[n, key] for n in names]
            ax.plot(xs, ys, color=col, linewidth=1.8, marker="o", markersize=4)
            ax.text(xs[-1] + 25, ys[-1], method, color=t["fg"], fontsize=8.5, va="center")
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xticks([0, 100, 200, 1000], ["0", "100", "200", "1000"])
        ax.set_xlim(-5, 1900)
        ax.set_title(title, loc="left", color=t["fg"], fontsize=9.5, pad=6)
        ax.set_xlabel("Training step", color=t["fg2"], fontsize=8.5)
        if fmt:
            ax.yaxis.set_major_formatter(fmt)
        ax.set_ylim(bottom=0, top=max(1.0, per[key].max() * 1.12) if fmt else per[key].max() * 1.12)
    fig.text(0.01, 0.005, "Measured on each model's own greedy answers to the same test words", color=t["fg2"],
             fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_entropy(t, ref):
    c = colors(t)
    names = ["base", "grpo-200", "opd-200", "opd-1000"]
    bins = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 10])
    r = ref[ref.model.isin(names)].assign(b=lambda d: np.digitize(d.t_entropy, bins) - 1)
    m = r.groupby(["model", "b"]).agg(x=("t_entropy", "mean"), y=("entropy", "mean"), n=("entropy", "size")).reset_index()
    m = m[m.n >= 30]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=t["bg"])
    style(ax, t)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    xt, yt = float(m.x.max()) * 1.08, float(m.y.max()) * 1.08
    ax.plot([0, xt], [0, xt], color=t["fg2"], linewidth=1, linestyle="--")
    ax.text(xt * 0.98, xt * 0.98 + 0.05, "same uncertainty as the teacher", color=t["fg2"], fontsize=7.5,
            ha="right", va="bottom")
    for n in names:
        d = m[m.model == n].sort_values("x")
        ax.plot(d.x, d.y, color=c[n], linewidth=1.8, marker="o", markersize=4)
    ax.set_xlim(0, xt)
    ax.set_ylim(0, yt)
    ax.set_xlabel("Teacher's entropy at that token (nats)", color=t["fg2"], fontsize=8.5)
    ax.set_ylabel("Model's entropy (nats)", color=t["fg2"], fontsize=8.5)
    ax.set_title("Uncertainty on the reference, binned by how unsure the teacher is", loc="left", color=t["fg"],
                 fontsize=10, pad=6)
    legend(fig, t, names, y=1.0, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def fig_examples(t, chars, words, k=6):
    names = ["base", "grpo-200", "opd-200", "teacher"]
    w = words.pivot_table(index="word_idx", columns="model", values="exact")
    lens = words[words.model == "base"].set_index("word_idx").reference.str.len()
    pick = w[(w["opd-200"] == 1) & (w["grpo-200"] == 0) & (w["base"] == 0)].index
    pick = [i for i in pick if 4 <= lens[i] <= 9][:k]
    if not pick:
        return None
    fig, axes = plt.subplots(len(pick), 1, figsize=(6.8, 1.05 * len(pick) + 0.6), facecolor=t["bg"])
    axes = np.atleast_1d(axes)
    cmap = mcolors.LinearSegmentedColormap.from_list("p", [t["bg"], t["series"][0]])
    for ax, wi in zip(axes, pick):
        ref = words[(words.model == "base") & (words.word_idx == wi)].iloc[0]
        cw = chars[(chars.word_idx == wi) & (chars.char_class != "end")]
        grid = np.array([[np.exp(cw[(cw.model == n) & (cw.char_idx == j)].logp.sum()) for j in range(len(ref.reference))]
                         for n in names])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for j, ch in enumerate(ref.reference):
            ax.text(j, -0.85, ch, ha="center", va="bottom", fontsize=10, color=t["fg"], family="Nirmala UI")
            for i in range(len(names)):
                ax.text(j, i, f"{grid[i, j]:.0%}" if grid[i, j] < 0.995 else "", ha="center", va="center",
                        fontsize=6.5, color=t["fg"])
        ax.set_yticks(range(len(names)), [LABEL[n].split(",")[0] for n in names], fontsize=7, color=t["fg2"])
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{ref.source_text}  →  GRPO wrote {words[(words.model == 'grpo-200') & (words.word_idx == wi)].prediction.iloc[0]}",
                     loc="left", fontsize=8, color=t["fg2"], pad=16, family=["DejaVu Sans", "Nirmala UI"])
    fig.text(0.01, 0.995, "P(correct letter) per model, words OPD-200 gets right and GRPO-200 doesn't (blank = ≥ 99.5%)",
             color=t["fg"], fontsize=8.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def summary(words, ref, greedy):
    g = greedy.assign(rejected=np.exp(greedy.t_logp) < 0.1)
    rows = []
    for n, w in words.groupby("model"):
        r, gg = ref[ref.model == n], g[g.model == n]
        rows.append({"model": n, "words": len(w), "exact": w.exact.mean(), "cer": w.cer.mean(),
                     "median_p_correct_word": float(np.median(np.exp(w.ref_logp))),
                     "ref_token_top1": r.top1.mean(), "ref_entropy": r.entropy.mean(), "ref_kl_to_teacher": r.kl.mean(),
                     "greedy_kl_to_teacher": gg.kl.mean(), "greedy_teacher_agree": gg.agree.mean(),
                     "greedy_teacher_rejected": gg.rejected.mean()})
    order = ["base", "grpo-100", "grpo-200", "opd-100", "opd-200", "opd-1000", "teacher"]
    return pd.DataFrame(rows).set_index("model").reindex([o for o in order if o in {r["model"] for r in rows}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/logprob-analysis")
    args = ap.parse_args()
    d = Path(args.data)
    ref, greedy, words = (pd.read_parquet(d / f) for f in ("tokens_ref.parquet", "tokens_greedy.parquet", "words.parquet"))
    chars = char_table(ref)
    s = summary(words, ref, greedy)
    s.to_csv(ROOT / "results" / "logprob_summary.csv", float_format="%.4f")
    print(s.round(3).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        for name, fn in (("word-prob", lambda: fig_word_prob(t, words)), ("delta", lambda: fig_delta(t, chars)),
                         ("by-class", lambda: fig_by_class(t, chars)), ("alignment", lambda: fig_alignment(t, greedy)),
                         ("entropy", lambda: fig_entropy(t, ref)), ("examples", lambda: fig_examples(t, chars, words))):
            fig = fn()
            if fig is None:
                continue
            path = OUT / f"logprobs-{name}-{mode}.png"
            fig.savefig(path, dpi=200, facecolor=t["bg"])
            plt.close(fig)
            print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
