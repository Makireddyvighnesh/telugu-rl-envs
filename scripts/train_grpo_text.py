"""GRPO + LoRA training on the Telugu text environment (transliteration by default).

Kaggle 2xT4 (one DDP process per GPU):

    accelerate launch --num_processes 2 scripts/train_grpo_text.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --out outputs/grpo-qwen15b-translit

Single GPU / quick check:

    python scripts/train_grpo_text.py --max-steps 5 --train-size 64

How a step works (GRPO):
  1. take a batch of words, generate --num-generations answers for each;
  2. score every answer with the environment's reward
     (0.8 * (1 - CER) + 0.2 * exact match);
  3. advantage = reward - mean reward of its group, so the model is pushed
     toward the answers that beat its own average and away from the rest;
  4. update only the LoRA adapter weights.

The reward is the same function the environment uses (telugu_envs.reward.text_reward),
called in-process; cer and exact are logged as zero-weight rewards for visibility.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_dataset(source: str, split: str, size: int | None, seed: int):
    from datasets import Dataset

    from telugu_envs.data import SOURCES
    from telugu_envs.text_env import TeluguTextEnvironment

    rows = []
    for s in SOURCES[source](split=split)(seed):
        rows.append(
            {
                "prompt": [{"role": "user", "content": TeluguTextEnvironment.prompt_for(s)}],
                "reference": s.reference,
                "task": s.meta.get("task", "transliteration"),
            }
        )
        if size and len(rows) >= size:
            break
    return Dataset.from_list(rows)


def _texts(completions) -> list[str]:
    # Conversational datasets give completions as [[{"role": "assistant", "content": ...}], ...]
    return [c[0]["content"] if isinstance(c, list) else c for c in completions]


def _scores(completions, reference, task):
    from telugu_envs.reward import text_reward

    return [text_reward(ref, pred, t) for pred, ref, t in zip(_texts(completions), reference, task)]


def env_reward(completions, reference, task, **kwargs) -> list[float]:
    return [s.reward for s in _scores(completions, reference, task)]


def one_minus_cer(completions, reference, task, **kwargs) -> list[float]:
    return [1.0 - s.cer for s in _scores(completions, reference, task)]


def exact_match(completions, reference, task, **kwargs) -> list[float]:
    return [float(s.exact) for s in _scores(completions, reference, task)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--source", default="aksharantar")
    ap.add_argument("--train-size", type=int, default=6000)
    ap.add_argument("--eval-size", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=32, help="completions per GPU per micro-step")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-completion-length", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in --out")
    ap.add_argument("--out", default="outputs/grpo-text")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    cuda = torch.cuda.is_available()
    out = Path(args.out)
    train = build_dataset(args.source, "train", args.train_size, args.seed)
    evals = build_dataset(args.source, "valid", args.eval_size, args.seed)

    config = GRPOConfig(
        output_dir=str(out),
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, args.max_steps // 20),
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=True,
        reward_weights=[1.0, 0.0, 0.0],  # only env_reward trains; the others are logged
        per_device_eval_batch_size=args.per_device_batch,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=5,
        log_completions=True,
        num_completions_to_print=4,
        bf16=False,  # TRL defaults to bf16, which T4s don't support
        fp16=cuda,  # T4: fp16 autocast; LoRA weights stay fp32
        use_cpu=not cuda,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=args.seed,
        model_init_kwargs={"dtype": torch.float16 if cuda else torch.float32},
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[env_reward, one_minus_cer, exact_match],
        args=config,
        train_dataset=train,
        eval_dataset=evals,
        peft_config=peft_config,
    )

    resume = args.resume and any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)

    if trainer.is_world_process_zero():
        trainer.save_model(str(out / "final"))
        history = trainer.state.log_history
        (out / "log_history.json").write_text(json.dumps(history, indent=2))
        rewards = [(h["step"], h["reward"]) for h in history if "reward" in h]
        evals_ = [(h["step"], h["eval_reward"]) for h in history if "eval_reward" in h]
        print("train reward (step, value):", rewards[:3], "...", rewards[-3:])
        print("eval reward (step, value):", evals_)
        print(f"adapter saved to {out / 'final'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
