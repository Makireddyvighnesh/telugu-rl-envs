"""A tiny synthetic snapshot for transport and scoring tests; never a benchmark."""

from .data.prepare import EVAL_SPLIT, write_snapshot

WORDS = [
    ("telugu", "తెలుగు", "Dakshina"),
    ("namaste", "నమస్తే", "AK-Freq"),
    ("hyderabad", "హైదరాబాద్", "AK-NEI"),
    ("sacramento", "శాక్రమెంటో", "AK-NEF"),
]


def make_fixture(directory):
    splits = {}
    for split in ("train", "validation", "test"):
        splits[split] = [
            {
                "task_id": f"te-translit/{split}/fx{i}",
                "split": split,
                "family": "transliteration",
                "origin": origin,
                "source_text": roman,
                "reference": native,
            }
            for i, (roman, native, origin) in enumerate(WORDS)
        ]
    splits[EVAL_SPLIT] = splits["validation"][:2]
    return write_snapshot(directory, splits, source="synthetic-fixture", revision="synthetic-fixture")
