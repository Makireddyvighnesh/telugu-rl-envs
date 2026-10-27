"""On-policy distillation (OPD) + LoRA on the Telugu text environment (transliteration by default).

Default: the GRPO-trained Qwen2.5-1.5B (teacher) into Qwen2.5-0.5B (student). Kaggle 2xT4, one process:
the student generates and trains on GPU 0, the teacher scores on GPU 1.

    python scripts/train_opd.py --out outputs/opd-qwen05b-translit

Quick check (CPU, or one GPU holding both models):

    python scripts/train_opd.py --max-steps 2 --train-size 16 --per-device-batch 2 --grad-accum 1

How a step works (OPD):
  1. take a batch of words, the student samples one answer for each;
  2. the teacher scores every token of the student's answer (its full next-token distribution);
  3. loss = per-token reverse KL(student || teacher) on the student's own tokens, so the student
     is pulled toward what the teacher would have written at each point it actually reached;
  4. update only the LoRA adapter weights.

Unlike GRPO there is no reward in the loss: a wrong answer still gets a dense per-token signal.
The environment reward (0.8 * (1 - CER) + 0.2 * exact match) of the sampled answers is logged
as env_reward / cer / exact so the curve is comparable with GRPO.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from train_grpo_text import build_dataset


def load_teacher(base: str, adapter: str | None, teacher_device, head_device, dtype):
    """Teacher = base + merged LoRA, backbone on `teacher_device` and an untied copy of `lm_head` on `head_device`.

    DistillationTrainer projects student and teacher hidden states through their lm_heads chunk by chunk, so the
    two heads must share a device; only the teacher's backbone forward runs on the other GPU.
    """
    import torch
    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    if adapter:
        from peft import PeftModel

        teacher = PeftModel.from_pretrained(teacher, adapter).merge_and_unload()
    teacher.to(teacher_device).eval().requires_grad_(False)
    if torch.device(teacher_device) != torch.device(head_device):
        w = teacher.lm_head.weight  # Qwen2.5 <= 3B ties lm_head to embed_tokens: copy it, don't move the embedding
        head = torch.nn.Linear(w.size(1), w.size(0), bias=False, device=head_device, dtype=w.dtype)
        head.weight.data.copy_(w.data)
        head.requires_grad_(False)
        teacher.lm_head = head
        teacher.config.tie_word_embeddings = False
        # A multi-device hf_device_map tells accelerate's prepare_model not to move the teacher back to GPU 0
        teacher.hf_device_map = {"model": str(teacher_device), "lm_head": str(head_device)}
    return teacher


def make_trainer_class():
    from trl import DistillationTrainer

    from telugu_envs.reward import text_reward

    class SplitDeviceDistillationTrainer(DistillationTrainer):
        """DistillationTrainer with the teacher backbone on its own GPU, logging the env reward of each rollout."""

        def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep, **kwargs):
            # Batches live on the student's device; feed each backbone on its own. The caller moves the returned
            # hidden states to the lm_head device.
            dev = unwrapped_model.get_input_embeddings().weight.device
            return super()._get_last_hidden_state(
                unwrapped_model, input_ids.to(dev), attention_mask.to(dev), logits_to_keep, **kwargs
            )

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            mode = "train" if self.model.training else "eval"
            ids = output["completion_ids"].masked_fill(output["completion_mask"] == 0, self._tokenizer.pad_token_id)
            preds = self._tokenizer.batch_decode(ids, skip_special_tokens=True)
            scores = [text_reward(x["reference"], p, x["task"]) for x, p in zip(inputs, preds)]
            for key, vals in (
                ("env_reward", [s.reward for s in scores]),
                ("cer", [s.cer for s in scores]),
                ("exact", [float(s.exact) for s in scores]),
            ):
                self._metrics[mode][key].append(sum(vals) / len(vals))
            return output

    return SplitDeviceDistillationTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="student")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct", help="teacher base model")
    ap.add_argument(
        "--teacher-adapter",
        default="makireddyvighnesh/Qwen2.5-1.5B-Instruct-telugu-translit-grpo",
        help="LoRA merged into the teacher ('' = base teacher)",
    )
    ap.add_argument("--source", default="aksharantar")
    ap.add_argument("--train-size", type=int, default=64000)
    ap.add_argument("--eval-size", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--per-device-batch", type=int, default=32, help="prompts (= completions) per micro-step")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling and loss temperature")
    ap.add_argument("--beta", type=float, default=1.0, help="0 = forward KL, 0.5 = JSD, 1 = reverse KL")
    ap.add_argument("--max-completion-length", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in --out")
    ap.add_argument("--out", default="outputs/opd-text")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig
    from trl import DistillationConfig

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32  # T4 has no bf16
    student_device = "cuda:0" if cuda else "cpu"
    teacher_device = "cuda:1" if cuda and torch.cuda.device_count() > 1 else student_device
    out = Path(args.out)
    train = build_dataset(args.source, "train", args.train_size, args.seed)
    evals = build_dataset(args.source, "valid", args.eval_size, args.seed)

    config = DistillationConfig(
        output_dir=str(out),
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, args.max_steps // 20),
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        temperature=args.temperature,
        beta=args.beta,
        max_completion_length=args.max_completion_length,
        per_device_eval_batch_size=args.per_device_batch,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=5,
        log_completions=True,
        num_completions_to_print=4,
        bf16=False,
        fp16=cuda,  # T4: fp16 autocast; LoRA weights stay fp32
        use_cpu=not cuda,
        report_to="none",
        seed=args.seed,
        model_init_kwargs={"dtype": dtype},
    )
    # One process drives both GPUs (student on 0, teacher on 1). Trainer would otherwise wrap the student in
    # nn.DataParallel across them, which DistillationTrainer refuses.
    if config.n_gpu > 1:
        config._n_gpu = 1

    teacher = load_teacher(args.teacher, args.teacher_adapter or None, teacher_device, student_device, dtype)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = make_trainer_class()(
        model=args.model,
        teacher_model=teacher,
        args=config,
        train_dataset=train,
        eval_dataset=evals,
        peft_config=peft_config,
    )
    print(f"student on {trainer.model.device}, teacher backbone on {teacher_device}, heads on {student_device}")

    resume = args.resume and any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)

    trainer.save_model(str(out / "final"))
    history = trainer.state.log_history
    (out / "log_history.json").write_text(json.dumps(history, indent=2))
    rewards = [(h["step"], round(h["env_reward"], 3)) for h in history if "env_reward" in h]
    evals_ = [(h["step"], round(h["eval_env_reward"], 3)) for h in history if "eval_env_reward" in h]
    print("train env_reward (step, value):", rewards[:3], "...", rewards[-3:])
    print("eval env_reward (step, value):", evals_)
    print(f"adapter saved to {out / 'final'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
