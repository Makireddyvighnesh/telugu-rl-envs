"""GRPO + LoRA that trains through the transliteration environment server.

    # 1. serve the environment (or point at the Hugging Face Space)
    TRANSLIT_SNAPSHOT=snapshot python -m telugu_transliteration.server.app &
    # 2. train against it (Kaggle 2xT4: one process per GPU)
    accelerate launch --num_processes 2 -m telugu_transliteration.train_grpo \
        --env-url http://127.0.0.1:8000 --out outputs/grpo-env

The dataset holds task IDs only. For every rollout TRL calls the environment's
``reset(task_id)``, which returns the prompt; the reward is the server's answer to
``step(completion)``. The hyperparameters match scripts/train_grpo_text.py (the
in-process run), and the default train words and eval split are the same words, so the
two runs are directly comparable.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-url", required=True, help="environment server or Space URL")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train-size", type=int, default=16000)
    ap.add_argument("--eval-split", default="validation-256")
    ap.add_argument("--eval-size", type=int, default=None, help="cap on eval tasks (default: the whole split)")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=32, help="completions per GPU per micro-step")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-completion-length", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--report-to", default="tensorboard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="outputs/grpo-env")
    args = ap.parse_args()

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from telugu_transliteration.client import connect
    from telugu_transliteration.training import TrainingEnvironment, env_cer, env_exact_match, env_reward, task_rows

    with connect(args.env_url) as client:
        manifest = client.manifest()
    snapshot_id = manifest["snapshot_id"]
    train = Dataset.from_list(task_rows(args.env_url, "train", limit=args.train_size, seed=args.seed))
    evals = Dataset.from_list(task_rows(args.env_url, args.eval_split, limit=args.eval_size))

    cuda = torch.cuda.is_available()
    out = Path(args.out)
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
        reward_weights=[1.0, 0.0, 0.0],  # only the environment's reward trains; 1-CER and exact are logged
        per_device_eval_batch_size=args.per_device_batch,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        log_completions=True,
        num_completions_to_print=4,
        bf16=False,  # T4s have no bf16
        fp16=cuda,
        use_cpu=not cuda,
        ddp_find_unused_parameters=False,
        report_to=args.report_to,
        seed=args.seed,
        model_init_kwargs={"dtype": torch.float16 if cuda else torch.float32},
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[env_reward, env_cer, env_exact_match],
        args=config,
        train_dataset=train,
        eval_dataset=evals,
        environment_factory=lambda: TrainingEnvironment(args.env_url, snapshot_id),
        peft_config=LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"
        ),
    )

    resume = args.resume and any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)

    if trainer.is_world_process_zero():
        trainer.save_model(str(out / "final"))
        history = trainer.state.log_history
        (out / "log_history.json").write_text(json.dumps(history, indent=2))
        (out / "environment.json").write_text(
            json.dumps({"env_url": args.env_url, "snapshot_id": snapshot_id, "revision": manifest.get("revision")}, indent=2)
        )
        evals_ = [(h["step"], round(h["eval_reward"], 4)) for h in history if "eval_reward" in h]
        print("eval reward (step, value):", evals_)
        print(f"adapter saved to {out / 'final'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
