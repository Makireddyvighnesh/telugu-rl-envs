"""Figures and summary numbers for the token-level GRPO vs OPD analysis (scripts/analyze_logprobs.py output).

    python scripts/plot_logprobs.py --data outputs/logprob-analysis

Writes results/logprob_summary.csv, results/graded_examples.json and three charts,
results/plots/logprobs-{learning,skipped,confidence}-{light,dark}.png. The two graded-answer figures
need real Telugu text shaping, which matplotlib lacks, so they are drawn in a browser from
graded_examples.json by scripts/graded_answers.html.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from analyze_logprobs import char_class  # noqa: E402
from plot_results import THEMES, style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "plots"
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")
STEPS = {"base": 0, "grpo-100": 100, "grpo-200": 200, "opd-100": 100, "opd-200": 200, "opd-1000": 1000}
X = {0: 0, 100: 1, 200: 2, 1000: 3.4}  # evenly spaced up to 200, then a gap to 1000
VIRAMA, ANUSVARA = "్", "ం"
EXAMPLES = {"intro": "sarkaaruku", "graded": ["alladam", "jaambavam", "padutunnaa", "holt"]}


def colors(t):
    return {"base": t["muted"], "grpo": t["series"][1], "opd": t["series"][0], "teacher": t["fg"]}


def summary(words, ref, greedy):
    g = greedy.assign(rejected=np.exp(greedy.t_logp) < 0.1)
    rows = []
    for n, w in words.groupby("model"):
        r, gg = ref[ref.model == n], g[g.model == n]
        rows.append({"model": n, "words": len(w), "exact": w.exact.mean(), "cer": w.cer.mean(),
                     "ref_token_top1": r.top1.mean(), "ref_entropy": r.entropy.mean(), "ref_kl_to_teacher": r.kl.mean(),
                     "greedy_kl_to_teacher": gg.kl.mean(), "greedy_teacher_agree": gg.agree.mean(),
                     "greedy_teacher_rejected": gg.rejected.mean(),
                     "answer_len_vs_ref": w.prediction.str.len().sum() / w.reference.str.len().sum(),
                     "virama_vs_ref": w.prediction.str.count(VIRAMA).sum() / w.reference.str.count(VIRAMA).sum(),
                     "anusvara_vs_ref": w.prediction.str.count(ANUSVARA).sum() / w.reference.str.count(ANUSVARA).sum()})
    order = ["base", "grpo-100", "grpo-200", "opd-100", "opd-200", "opd-1000", "teacher"]
    return pd.DataFrame(rows).set_index("model").reindex([o for o in order if o in {r["model"] for r in rows}])


def fig_learning(t, s):
    """Exact match and distance to the teacher over training, GRPO vs OPD from the same 0.5B base."""
    c = colors(t)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), facecolor=t["bg"])
    panels = [("exact", "Words spelled exactly right", PCT, s.loc["teacher", "exact"]),
              ("greedy_kl_to_teacher", "How differently from the teacher it writes", None, 0.0)]
    for ax, (key, title, fmt, teacher) in zip(axes, panels):
        style(ax, t)
        ax.axhline(teacher, color=c["teacher"], linewidth=1.2, linestyle="--")
        lbl = f"teacher (GRPO 1.5B)  {teacher:.0%}" if fmt else "teacher = 0"
        ax.text(4.55, teacher, lbl, color=t["fg"], fontsize=8.5, va="bottom", ha="right")
        for method, names in (("GRPO", ["base", "grpo-100", "grpo-200"]), ("OPD", ["base", "opd-100", "opd-200", "opd-1000"])):
            xs = [X[STEPS[n]] for n in names]
            ys = [s.loc[n, key] for n in names]
            col = c[method.lower()]
            ax.plot(xs[:3], ys[:3], color=col, linewidth=2, marker="o", markersize=5)
            if len(xs) > 3:
                ax.plot(xs[2:], ys[2:], color=col, linewidth=2, marker="o", markersize=5, linestyle=(0, (2, 2)))
            val = f"{ys[-1]:.0%}" if fmt else f"{ys[-1]:.2f}"
            ax.text(xs[-1] + 0.12, ys[-1], f"{method}  {val}", color=col, fontsize=9, va="center", weight="bold")
        ax.set_xticks([X[k] for k in (0, 100, 200, 1000)], ["start", "100", "200", "1000"])
        ax.set_xlim(-0.2, 4.6)
        ax.set_xlabel("Training steps (same 0.5B base model)", color=t["fg2"], fontsize=8.5)
        ax.set_title(title, loc="left", color=t["fg"], fontsize=10.5, pad=8)
        if fmt:
            ax.yaxis.set_major_formatter(fmt)
            ax.set_ylim(0, max(0.32, teacher * 1.12))
        else:
            ax.set_ylim(-0.12, s.loc["base", key] * 1.1)
            ax.set_ylabel("nats per token (reverse KL)", color=t["fg2"], fontsize=8)
    axes[1].text(0.1, s.loc["base", "greedy_kl_to_teacher"] * 1.02, "lower = closer to the teacher", color=t["fg2"],
                 fontsize=7.5, va="bottom")
    fig.tight_layout()
    return fig


def fig_skipped(t, s):
    """Letters the answers contain, relative to the correct answers (100% = same as the references)."""
    c = colors(t)
    models = [("base", "Base 0.5B", c["base"]), ("grpo-200", "GRPO, 200 steps", c["grpo"]),
              ("opd-200", "OPD, 200 steps", c["opd"]), ("teacher", "Teacher (1.5B)", c["teacher"])]
    panels = [("virama_vs_ref", "Conjunct marks (virama)"), ("anusvara_vs_ref", "Anusvara (nasal ending)"), ("answer_len_vs_ref", "Letters per answer")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), facecolor=t["bg"], sharey=True)
    for ax, (key, title) in zip(axes, panels):
        style(ax, t)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=t["grid"], linewidth=0.8)
        vals = [s.loc[m, key] for m, _, _ in models]
        ax.barh(range(len(models)), vals, height=0.62, color=[col for _, _, col in models])
        for i, v in enumerate(vals):
            ax.text(v + 0.03, i, f"{v:.0%}", va="center", fontsize=9, color=t["fg"])
        ax.axvline(1.0, color=t["fg2"], linewidth=1, linestyle="--")
        ax.set_xlim(0, 1.6)
        ax.xaxis.set_major_formatter(PCT)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.set_title(title, loc="left", color=t["fg"], fontsize=10.5, pad=8, family=["DejaVu Sans", "Nirmala UI"])
    axes[0].set_yticks(range(len(models)), [n for _, n, _ in models], color=t["fg"], fontsize=9)
    axes[0].invert_yaxis()
    fig.text(0.5, 0.02, "Count in the model's answers ÷ count in the correct answers, 2,000 test words  ·  "
             "dashed line = same as the correct answers", color=t["fg2"], fontsize=8, ha="center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def fig_confidence(t, s):
    """Uncertainty (entropy) against how often the first guess is right, with each method's path."""
    c = colors(t)
    fig, ax = plt.subplots(figsize=(6.6, 4.3), facecolor=t["bg"])
    style(ax, t)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    for method, names in (("grpo", ["base", "grpo-100", "grpo-200"]), ("opd", ["base", "opd-100", "opd-200", "opd-1000"])):
        xs = [s.loc[n, "ref_entropy"] for n in names]
        ys = [s.loc[n, "ref_token_top1"] for n in names]
        for i in range(len(xs) - 1):
            ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="-|>", color=c[method], lw=2, shrinkA=5, shrinkB=5, mutation_scale=12))
        offs = {"grpo-100": (0, -14), "grpo-200": (10, 4), "opd-100": (-6, -14), "opd-200": (14, -10), "opd-1000": (12, -4)}
        for n, x, y in zip(names[1:], xs[1:], ys[1:]):
            ax.plot(x, y, "o", color=c[method], markersize=6)
            ax.annotate(f"{STEPS[n]}", (x, y), xytext=offs[n], textcoords="offset points", color=t["fg2"], fontsize=7.5,
                        ha="center", va="center")
    bx, by = s.loc["base", "ref_entropy"], s.loc["base", "ref_token_top1"]
    tx, ty = s.loc["teacher", "ref_entropy"], s.loc["teacher", "ref_token_top1"]
    ax.plot(bx, by, "o", color=c["base"], markersize=8)
    ax.text(bx, by + 0.015, "Base 0.5B", color=t["fg"], fontsize=9, ha="center", va="bottom")
    ax.plot(tx, ty, "*", color=c["teacher"], markersize=13)
    ax.text(tx + 0.04, ty + 0.012, "Teacher", color=t["fg"], fontsize=9, va="bottom")
    g2 = s.loc["grpo-200"]
    ax.text(g2.ref_entropy + 0.12, g2.ref_token_top1 - 0.065, "GRPO: more confident,\nnot more correct",
            color=c["grpo"], fontsize=9, weight="bold", va="top")
    o2 = s.loc["opd-200"]
    ax.text(o2.ref_entropy + 0.62, o2.ref_token_top1 + 0.03, "OPD: more confident\nand more correct,\nclose to the teacher",
            color=c["opd"], fontsize=9, weight="bold", va="center")
    ax.set_xlim(1.45, -0.05)  # reversed: right = more confident
    ax.set_ylim(0.55, 0.9)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Uncertainty (entropy, nats)   ← less sure  ·  more sure →", color=t["fg2"], fontsize=8.5)
    ax.set_ylabel("Next-token guesses that are correct", color=t["fg2"], fontsize=8.5)
    ax.set_title("Becoming confident vs becoming correct", loc="left", color=t["fg"], fontsize=10.5, pad=8)
    fig.text(0.01, 0.01, "Teacher-forced on the correct answers of 2,000 test words; numbers are training steps",
             color=t["fg2"], fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def syllables(s: str):
    """Split Telugu text into syllables: signs and the consonant after a virama attach to the previous one."""
    out = []
    for i, ch in enumerate(s):
        k = char_class(ch)
        if out and (k in ("vowel sign", "virama", "nasal/visarga sign") or (i and s[i - 1] == VIRAMA and k == "consonant")):
            out[-1].append(i)
        else:
            out.append([i])
    return out


def graded_examples(words, greedy, tokenizer):
    """Each model's greedy answer per syllable, with the teacher's penalty (log p_student - log p_teacher)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    def grade(model, wi):
        rows = greedy[(greedy.model == model) & (greedy.word_idx == wi)].sort_values("pos")
        ids, pen = rows.token_id.tolist(), (rows.logp - rows.t_logp).tolist()
        stop = None
        if ids and ids[-1] == eos:
            stop, ids, pen = pen[-1], ids[:-1], pen[:-1]
        text = tok.decode(ids)
        b2c = [ci for ci, ch in enumerate(text) for _ in ch.encode("utf-8")]
        cpen, pos = [0.0] * len(text), 0
        for i, p in zip(ids, pen):
            pos += len(tok.convert_ids_to_tokens(i))
            cpen[b2c[min(pos, len(b2c)) - 1]] += p
        return {"syllables": [{"text": "".join(text[j] for j in cl), "penalty": round(sum(cpen[j] for j in cl), 2)}
                              for cl in syllables(text)],
                "stop": None if stop is None else round(stop, 2)}

    by_src = words[words.model == "base"].set_index("source_text")
    out = {"intro": None, "graded": []}
    for key in [EXAMPLES["intro"]] + EXAMPLES["graded"]:
        wi = int(by_src.loc[key, "word_idx"])
        entry = {"source": key, "reference": by_src.loc[key, "reference"],
                 "reference_syllables": ["".join(by_src.loc[key, "reference"][j] for j in cl)
                                         for cl in syllables(by_src.loc[key, "reference"])], "answers": {}}
        for m in ("base", "grpo-200", "opd-200"):
            w = words[(words.model == m) & (words.word_idx == wi)].iloc[0]
            entry["answers"][m] = {"text": w.prediction, "exact": bool(w.exact), "reward": round(float(w.reward), 2),
                                   **grade(m, wi)}
        if key == EXAMPLES["intro"]:
            out["intro"] = entry
        else:
            out["graded"].append(entry)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/logprob-analysis")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()
    d = Path(args.data)
    ref, greedy, words = (pd.read_parquet(d / f) for f in ("tokens_ref.parquet", "tokens_greedy.parquet", "words.parquet"))
    s = summary(words, ref, greedy)
    s.to_csv(RESULTS / "logprob_summary.csv", float_format="%.4f")
    print(s.round(3).to_string())
    ex = graded_examples(words, greedy, args.tokenizer)
    (RESULTS / "graded_examples.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote results/graded_examples.json")
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        for name, fn in (("learning", fig_learning), ("skipped", fig_skipped), ("confidence", fig_confidence)):
            fig = fn(t, s)
            path = OUT / f"logprobs-{name}-{mode}.png"
            fig.savefig(path, dpi=200, facecolor=t["bg"])
            plt.close(fig)
            print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
