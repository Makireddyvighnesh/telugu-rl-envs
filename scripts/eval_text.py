"""Baseline evaluation for the Telugu text environment (transliteration / translation).

Runs one model replica per GPU (data parallel), each on a shard of the data,
then merges and scores with the environment's reward.

    python scripts/eval_text.py --model Qwen/Qwen2.5-0.5B-Instruct --source aksharantar \
        --split test --replicas 2 --out outputs/qwen05b-translit-test

Two measurements:
  greedy   : one deterministic answer per sample -> the baseline score.
  sampling : k answers per sample at temperature T on a subset -> how much
             GRPO can learn. GRPO's advantage is (reward - group mean), so a
             group where all k answers score the same gives zero signal.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch.multiprocessing as mp


def load_samples(source: str, split: str, limit: int | None):
    from telugu_envs.data import SOURCES

    samples = list(SOURCES[source](split=split)(None))
    return samples[:limit] if limit else samples


def worker(rank: int, args, shard: list[dict], k: int, temperature: float, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"  # CPU only for local smoke tests
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if device == "cuda" else torch.float32  # T4 has no bf16
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to(device).eval()

    outputs = []
    for i in range(0, len(shard), args.batch_size):
        batch = shard[i : i + args.batch_size]
        texts = [
            tok.apply_chat_template([{"role": "user", "content": s["prompt"]}], tokenize=False, add_generation_prompt=True)
            for s in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if k == 1:
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, num_return_sequences=k)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        out = out[:, enc["input_ids"].shape[1] :]
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        for j, s in enumerate(batch):
            outputs.append({**s, "predictions": decoded[j * k : (j + 1) * k]})
        if rank == 0 and (i // args.batch_size) % 10 == 0:
            print(f"[gpu{rank}] k={k} {i + len(batch)}/{len(shard)}", flush=True)
    queue.put(outputs)


def run_parallel(args, items: list[dict], k: int, temperature: float) -> list[dict]:
    shards = [items[r :: args.replicas] for r in range(args.replicas)]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, args, shards[r], k, temperature, queue)) for r in range(args.replicas)]
    for p in procs:
        p.start()
    results = [x for _ in procs for x in queue.get()]
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker exited with {p.exitcode}")
    return results


def score(results: list[dict]) -> list[dict]:
    from telugu_envs.reward import text_reward

    for r in results:
        r["scores"] = [vars(text_reward(r["reference"], p, r["task"])) for p in r["predictions"]]
    return results


def summarize_greedy(results):
    def agg(rows):
        s = [r["scores"][0] for r in rows]
        return {
            "n": len(rows),
            "reward": st.mean(x["reward"] for x in s),
            "cer": st.mean(x["cer"] for x in s),
            "exact_match": st.mean(x["exact"] for x in s),
            "chrf": st.mean(x["chrf"] for x in s),
            "flagged": sum(bool(x["flags"]) for x in s),
        }

    by_origin = defaultdict(list)
    for r in results:
        by_origin["+".join(r["origins"])].append(r)
    return {"overall": agg(results), "by_origin": {k: agg(v) for k, v in sorted(by_origin.items())}}


def summarize_sampling(results):
    groups = [[x["reward"] for x in r["scores"]] for r in results]
    spreads = [max(g) - min(g) for g in groups]
    return {
        "groups": len(groups),
        "k": len(groups[0]) if groups else 0,
        "mean_reward": st.mean(x for g in groups for x in g),
        "best_of_k_reward": st.mean(max(g) for g in groups),
        "pass_at_k_exact": st.mean(any(x["exact"] for x in r["scores"]) for r in results),
        "groups_with_signal": st.mean(s > 1e-9 for s in spreads),
        "all_zero_groups": st.mean(max(g) == 0 for g in groups),
        "all_perfect_groups": st.mean(min(g) == 1 for g in groups),
        "mean_within_group_std": st.mean(st.pstdev(g) for g in groups),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (e.g. a GRPO output's final/)")
    ap.add_argument("--source", default="aksharantar")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="cap on greedy-eval samples")
    ap.add_argument("--replicas", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--k", type=int, default=8, help="samples per prompt for the GRPO-signal check (0 = skip)")
    ap.add_argument("--k-subset", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    from telugu_envs.text_env import TeluguTextEnvironment

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.source, args.split, args.limit)
    items = [
        {
            "id": s.id,
            "prompt": TeluguTextEnvironment.prompt_for(s),
            "source_text": s.text,
            "reference": s.reference,
            "task": s.meta.get("task"),
            "origins": s.meta.get("origins", [s.meta.get("source")]),
        }
        for s in samples
    ]
    print(f"{len(items)} samples from {args.source}/{args.split}; model={args.model}; replicas={args.replicas}", flush=True)

    report = {"model": args.model, "adapter": args.adapter, "source": args.source, "split": args.split, "replicas": args.replicas}
    t = time.time()
    greedy = score(run_parallel(args, items, k=1, temperature=0.0))
    report["greedy"] = summarize_greedy(greedy)
    report["greedy_seconds"] = round(time.time() - t, 1)
    (out / "greedy_predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in greedy))
    print(json.dumps(report["greedy"], indent=2), flush=True)

    if args.k > 1:
        t = time.time()
        subset = items[:: max(1, len(items) // args.k_subset)][: args.k_subset]
        sampled = score(run_parallel(args, subset, k=args.k, temperature=args.temperature))
        report["sampling"] = {**summarize_sampling(sampled), "temperature": args.temperature}
        report["sampling_seconds"] = round(time.time() - t, 1)
        (out / "sampled_predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sampled))
        print(json.dumps(report["sampling"], indent=2), flush=True)

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out}/report.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
