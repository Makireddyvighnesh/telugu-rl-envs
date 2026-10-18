"""A prepared snapshot of transliteration tasks. References never enter a public payload.

A snapshot directory holds ``manifest.json`` and one ``<split>.jsonl`` per split. Each
row is one task: ``task_id``, ``split``, ``family``, ``origin``, ``source_text`` and the
private ``reference``. ``data/prepare.py`` builds it from AI4Bharat Aksharantar.
"""

import json
from pathlib import Path

SPLITS = ("train", "validation", "test")
FAMILY = "transliteration"

PROMPT = (
    "Transliterate the following romanized Telugu into Telugu script. "
    "Output only the Telugu text.\n\n{text}"
)

PUBLIC_FIELDS = ("task_id", "split", "family", "origin", "source_text", "prompt")


class Catalog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.snapshot_id = self.manifest["snapshot_id"]
        self._splits: dict[str, list[dict]] = {}
        self._by_id: dict[str, dict] = {}
        for name in self.manifest["splits"]:
            rows = []
            with (self.directory / f"{name}.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        task = json.loads(line)
                        task["prompt"] = PROMPT.format(text=task["source_text"])
                        rows.append(task)
                        self._by_id[task["task_id"]] = task
            self._splits[name] = rows

    def splits(self):
        return list(self._splits)

    def count(self, split):
        if split not in self._splits:
            raise ValueError(f"Unknown split {split!r}; expected one of {self.splits()}")
        return len(self._splits[split])

    def at(self, split, index):
        count = self.count(split)
        if not 0 <= index < count:
            raise IndexError(f"{split} has {count} tasks; index {index} is out of range")
        return self._splits[split][index]

    def get(self, task_id):
        try:
            return self._by_id[task_id]
        except KeyError:
            raise KeyError(f"Unknown task_id {task_id!r}") from None

    def public(self, task):
        return {key: task[key] for key in PUBLIC_FIELDS} | {"snapshot_id": self.snapshot_id}

    def task_range(self, split, start=None, stop=None):
        count = self.count(split)
        start = 0 if start is None else max(0, start)
        stop = count if stop is None else min(stop, count)
        return [self.public(task) | {"index": i} for i, task in zip(range(start, stop), self._splits[split][start:stop])]
