"""Token-level analysis: how GRPO and OPD changed the student's next-token distributions.

For every model and every sampled test word, two passes, each scored against the teacher:

  reference pass   teacher-forced on the correct Telugu answer (+ <|im_end|>). Per token: the model's
                   log-prob of the correct token, whether it is the model's top choice, its entropy, and
                   the reverse KL(model || teacher) of the full next-token distribution at that point.
  greedy pass      the model's own greedy answer. Per token: the model's and the teacher's log-prob of
                   the chosen token, whether the teacher's top choice agrees, and reverse KL(model ||
                   teacher). This is the quantity OPD minimizes, measured on the model's own rollouts.

Qwen splits Telugu at the byte level (1-2 tokens per letter), so every reference token is also mapped to
the Telugu character that contains its last byte, with that character's class (consonant, vowel sign,
virama, ...), so results can be read per letter.

Kaggle 2xT4 (student on GPU 0, teacher on GPU 1):

    python scripts/analyze_logprobs.py --n 2000 --out outputs/logprob-analysis \\
        --model base=Qwen/Qwen2.5-0.5B-Instruct \\
        --model grpo-200=Qwen/Qwen2.5-0.5B-Instruct:/path/to/grpo/final ...

Writes tokens_ref.parquet, tokens_greedy.parquet and words.parquet to --out.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_TEACHER = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TEACHER_ADAPTER = "makireddyvighnesh/Qwen2.5-1.5B-Instruct-telugu-translit-grpo"


def char_class(c: str) -> str:
    o = ord(c)
    if 0x0C00 <= o <= 0x0C04:
        return "nasal/visarga sign"
    if 0x0C05 <= o <= 0x0C14 or 0x0C60 <= o <= 0x0C61:
        return "vowel"
    if 0x0C15 <= o <= 0x0C39 or 0x0C58 <= o <= 0x0C5A:
        return "consonant"
    if 0x0C3E <= o <= 0x0C4C or o in (0x0C55, 0x0C56, 0x0C62, 0x0C63):
        return "vowel sign"
    if o == 0x0C4D:
        return "virama"
    return "other"


def load_model(spec: str, dtype, device):
    import torch
    from transformers import AutoModelForCausalLM

    base, _, adapter = spec.partition(":")
    # A Windows drive letter ("C:/...") is not an adapter separator.
    if len(base) == 1 and adapter:
        base, _, adapter = f"{base}:{adapter}".partition(":")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.to(device).eval().requires_grad_(False), torch


def build_items(tok, n: int, split: str):
    from telugu_envs.data import SOURCES
    from telugu_envs.text_env import TeluguTextEnvironment

    samples = list(SOURCES["aksharantar"](split=split)(None))
    stride = max(1, len(samples) // n)
    samples = samples[::stride][:n]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    byte_len = lambda i: len(tok.convert_ids_to_tokens(i))  # byte-level BPE: one symbol per byte
    items = []
    for idx, s in enumerate(samples):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": TeluguTextEnvironment.prompt_for(s)}], tokenize=False, add_generation_prompt=True
        )
        ref_ids = tok(s.reference, add_special_tokens=False).input_ids
        byte_to_char = [ci for ci, ch in enumerate(s.reference) for _ in ch.encode("utf-8")]
        pos, chars = 0, []
        for i in ref_ids:
            pos += byte_len(i)
            ci = byte_to_char[min(pos, len(byte_to_char)) - 1]
            chars.append((ci, char_class(s.reference[ci])))
        chars.append((len(s.reference), "end"))
        items.append(
            {
                "word_idx": idx,
                "id": s.id,
                "origin": "+".join(s.meta.get("origins", [s.meta.get("source", "")])),
                "source_text": s.text,
                "reference": s.reference,
                "task": s.meta.get("task", "transliteration"),
                "prompt_ids": tok(prompt, add_special_tokens=False).input_ids,
                "ref_ids": ref_ids + [im_end],
                "ref_chars": chars,
            }
        )
    return items


def score_sequences(torch, student, teacher, prompts, targets, pad_id):
    """Left-pad prompt+target, run both models, return per-target-token stats for each sequence."""
    seqs = [p + t for p, t in zip(prompts, targets)]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s) :] = torch.tensor(s)
        mask[i, L - len(s) :] = 1
    pos_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    # Row i's target tokens sit in the last len(targets[i]) columns; they are predicted one column earlier.
    sel = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, t in enumerate(targets):
        sel[i, L - len(t) - 1 : L - 1] = True
    tgt = torch.cat([torch.tensor(t) for t in targets])

    def logprobs(model):
        dev = model.lm_head.weight.device
        h = model.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos_ids.to(dev)).last_hidden_state
        return torch.log_softmax(model.lm_head(h[sel.to(dev)]).float(), dim=-1)

    lp_s = logprobs(student)
    lp_t = logprobs(teacher).to(lp_s.device)
    tgt = tgt.to(lp_s.device)
    p_s = lp_s.exp()
    out = {
        "logp": lp_s.gather(1, tgt[:, None])[:, 0],
        "t_logp": lp_t.gather(1, tgt[:, None])[:, 0],
        "rank": (lp_s > lp_s.gather(1, tgt[:, None])).sum(1),
        "top1": lp_s.argmax(1) == tgt,
        "t_top1": lp_t.argmax(1) == tgt,
        "agree": lp_s.argmax(1) == lp_t.argmax(1),
        "entropy": -(p_s * lp_s).sum(1),
        "t_entropy": -(lp_t.exp() * lp_t).sum(1),
        "kl": (p_s * (lp_s - lp_t)).sum(1),
    }
    out = {k: v.float().cpu().tolist() for k, v in out.items()}
    rows, k = [], 0
    for t in targets:
        rows.append({key: vals[k : k + len(t)] for key, vals in out.items()})
        k += len(t)
    return rows


def greedy(torch, model, prompts, pad_id, eos_id, max_new):
    L = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), L), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, p in enumerate(prompts):
        ids[i, L - len(p) :] = torch.tensor(p)
        mask[i, L - len(p) :] = 1
    dev = model.lm_head.weight.device
    out = model.generate(
        input_ids=ids.to(dev), attention_mask=mask.to(dev), max_new_tokens=max_new, do_sample=False,
        pad_token_id=pad_id, eos_token_id=eos_id,
    )[:, L:].cpu().tolist()
    gens = []
    for g in out:
        if eos_id in g:
            g = g[: g.index(eos_id) + 1]  # keep the stop token: choosing to stop is part of the answer
        gens.append([t for t in g if t != pad_id] or [eos_id])
    return gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=base_model[:adapter_path]; repeatable")
    ap.add_argument("--teacher", default=DEFAULT_TEACHER)
    ap.add_argument("--teacher-adapter", default=DEFAULT_TEACHER_ADAPTER, help="'' = base teacher")
    ap.add_argument("--n", type=int, default=2000, help="test words, evenly strided over the split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default="outputs/logprob-analysis")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from transformers import AutoTokenizer

    from telugu_envs.reward import text_reward

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    s_dev = "cuda:0" if cuda else "cpu"
    t_dev = "cuda:1" if cuda and torch.cuda.device_count() > 1 else s_dev
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.teacher)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    items = build_items(tok, args.n, args.split)
    print(f"{len(items)} words from aksharantar/{args.split}", flush=True)

    teacher_spec = args.teacher + (f":{args.teacher_adapter}" if args.teacher_adapter else "")
    teacher, _ = load_model(teacher_spec, dtype, t_dev)
    specs = [m.split("=", 1) for m in args.model] + [["teacher", teacher_spec]]

    ref_rows, greedy_rows, word_rows = [], [], []
    for name, spec in specs:
        t0 = time.time()
        student = teacher if name == "teacher" else load_model(spec, dtype, s_dev)[0]
        for b in range(0, len(items), args.batch_size):
            batch = items[b : b + args.batch_size]
            prompts = [it["prompt_ids"] for it in batch]
            with torch.no_grad():
                ref = score_sequences(torch, student, teacher, prompts, [it["ref_ids"] for it in batch], pad_id)
                gens = greedy(torch, student, prompts, pad_id, eos_id, args.max_new_tokens)
                gsc = score_sequences(torch, student, teacher, prompts, gens, pad_id)
            for it, r, g, gs in zip(batch, ref, gens, gsc):
                for p, (ci, cls) in enumerate(it["ref_chars"]):
                    ref_rows.append({"model": name, "word_idx": it["word_idx"], "pos": p, "n_tok": len(it["ref_ids"]),
                                     "char_idx": ci, "char_class": cls, **{k: v[p] for k, v in r.items()}})
                for p in range(len(g)):
                    greedy_rows.append({"model": name, "word_idx": it["word_idx"], "pos": p, "token_id": g[p],
                                        **{k: gs[k][p] for k in ("logp", "t_logp", "agree", "entropy", "t_entropy", "kl")}})
                pred = tok.decode(g, skip_special_tokens=True)
                sc = text_reward(it["reference"], pred, it["task"])
                word_rows.append({"model": name, "word_idx": it["word_idx"], "id": it["id"], "origin": it["origin"],
                                  "source_text": it["source_text"], "reference": it["reference"], "prediction": pred,
                                  "reward": sc.reward, "cer": sc.cer, "exact": bool(sc.exact),
                                  "ref_logp": sum(r["logp"]), "ref_t_logp": sum(r["t_logp"]),
                                  "ref_kl_mean": sum(r["kl"]) / len(r["kl"]), "greedy_kl_mean": sum(gs["kl"]) / len(gs["kl"]),
                                  "greedy_t_logp": sum(gs["t_logp"]), "greedy_len": len(g)})
        if name != "teacher":
            del student
            if cuda:
                torch.cuda.empty_cache()
        w = [r for r in word_rows if r["model"] == name]
        print(f"[{name}] {time.time() - t0:.0f}s  exact {sum(r['exact'] for r in w) / len(w):.3f}  "
              f"mean ref logp {sum(r['ref_logp'] for r in w) / len(w):.2f}  "
              f"greedy KL to teacher {sum(r['greedy_kl_mean'] for r in w) / len(w):.3f}", flush=True)

    pd.DataFrame(ref_rows).to_parquet(out / "tokens_ref.parquet", index=False)
    pd.DataFrame(greedy_rows).to_parquet(out / "tokens_greedy.parquet", index=False)
    pd.DataFrame(word_rows).to_parquet(out / "words.parquet", index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
