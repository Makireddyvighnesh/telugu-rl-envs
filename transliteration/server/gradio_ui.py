"""Playground: pick a word, type its Telugu, see the score and only then the reference.

The OpenEnv discovery and observation APIs never return a reference; this deliberately
reveals it after grading, which is what makes the playground useful for judging whether
the reward is fair.
"""

import random

import gradio as gr

from ..models import TransliterationAction
from .environment import TransliterationEnvironment


class Playground:
    def __init__(self, catalog):
        self.catalog = catalog

    def choose(self, split, current=None, direction=0):
        count = self.catalog.count(split)
        if not count:
            return "", "", "No tasks in this split.", "", "", ""
        index = random.randrange(count)
        if direction and current:
            task = self.catalog.get(current)
            if task["split"] == split:
                index = (self.catalog._splits[split].index(task) + direction) % count
        task = self.catalog.at(split, index)
        details = f"origin **{task['origin']}** · task `{task['task_id']}`"
        return task["task_id"], task["source_text"], details, f"{index + 1:,} / {count:,}", "", ""

    def submit(self, task_id, answer):
        if not task_id:
            return "Select a task first.", ""
        environment = TransliterationEnvironment(self.catalog)
        environment.reset(task_id=task_id)
        result = environment.step(TransliterationAction(answer=answer or ""))
        m = result.metrics
        parts = [f"**Reward {result.reward:.3f}**", f"CER {m['cer']:.3f}"]
        if m.get("exact_match"):
            parts.append("exact match")
        if m.get("empty"):
            parts.append("empty answer")
        if m.get("degenerate"):
            parts.append("repetitive answer, scored 0")
        return " · ".join(parts), self.catalog.get(task_id)["reference"]


def build_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    catalog = web_manager.env.catalog
    playground = Playground(catalog)
    splits = [s for s in catalog.splits() if catalog.count(s)]
    total = sum(catalog.count(s) for s in splits)

    with gr.Blocks(title="Telugu transliteration") as demo:
        gr.Markdown(
            "# Telugu transliteration · Aksharantar\n"
            "Read a romanized Telugu word, write it in Telugu script, and see how the reward is "
            "computed: `0.8 × (1 − CER) + 0.2 × exact match`.\n\n"
            f"**{total:,} tasks** from human-romanized Aksharantar sources."
        )
        with gr.Row():
            split = gr.Dropdown(splits, value=splits[0], label="Split", scale=2)
            previous = gr.Button("← Previous")
            shuffle = gr.Button("Shuffle", variant="primary")
            following = gr.Button("Next →")
            position = gr.Textbox(label="Position", interactive=False, scale=1)
        details = gr.Markdown()
        source = gr.Textbox(label="Romanized Telugu", interactive=False)
        answer = gr.Textbox(label="Your answer in Telugu script", placeholder="తెలుగు")
        grade = gr.Button("Submit", variant="primary")
        score = gr.Markdown()
        reference = gr.Textbox(label="Reference (revealed after scoring)", interactive=False)
        task_id = gr.State("")

        outputs = [task_id, source, details, position, score, reference]
        split.change(lambda s: playground.choose(s), [split], outputs)
        shuffle.click(lambda s: playground.choose(s), [split], outputs)
        previous.click(lambda s, c: playground.choose(s, c, -1), [split, task_id], outputs)
        following.click(lambda s, c: playground.choose(s, c, 1), [split, task_id], outputs)
        grade.click(playground.submit, [task_id, answer], [score, reference])
        answer.submit(playground.submit, [task_id, answer], [score, reference])
        demo.load(lambda s: playground.choose(s), [split], outputs)
    return demo
