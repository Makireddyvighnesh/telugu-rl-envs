"""Build a snapshot from AI4Bharat Aksharantar (Telugu).

    python -m telugu_transliteration.data.prepare --output snapshot

Keeps the human-romanized sources only (Dakshina, AK-Freq, AK-NEI, AK-NEF). Telugu
train is otherwise 95% IndicCorp: web-mined, model-romanized, and about 1 in 8 native
words misspelled in a 40-row audit. That leaves 50.9k train, 3.7k validation and 10.3k
test pairs. ``validation-256`` is a frozen eval split: the first 256 validation tasks
after a seed-0 shuffle, the same words the in-process GRPO runs evaluated on.
"""

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path

REPO = "ai4bharat/Aksharantar"
ARCHIVE = "tel.zip"
HUMAN_SOURCES = ("Dakshina", "AK-Freq", "AK-NEI", "AK-NEF")
FILE_SPLITS = {"train": "train", "validation": "valid", "test": "test"}
EVAL_SPLIT, EVAL_SIZE, EVAL_SEED = "validation-256", 256, 0


def read_split(archive, file_split):
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith(f"_{file_split}.json")]
        if len(names) != 1:
            raise ValueError(f"Expected one *_{file_split}.json in {archive}, found {names}")
        with z.open(names[0]) as handle:
            return [json.loads(line) for line in handle if line.strip()]


def build_tasks(archive, sources=HUMAN_SOURCES):
    splits, skipped = {}, {}
    for split, file_split in FILE_SPLITS.items():
        rows, seen = [], set()
        for row in read_split(archive, file_split):
            roman, native, origin = row.get("english word"), row.get("native word"), row.get("source")
            if not roman or not native:
                skipped["missing_text"] = skipped.get("missing_text", 0) + 1
                continue
            if sources is not None and origin not in sources:
                skipped["machine_romanized_source"] = skipped.get("machine_romanized_source", 0) + 1
                continue
            uid = str(row.get("unique_identifier") or len(rows))
            task_id = f"te-translit/{split}/{uid}"
            if task_id in seen:
                skipped["duplicate_id"] = skipped.get("duplicate_id", 0) + 1
                continue
            seen.add(task_id)
            rows.append(
                {
                    "task_id": task_id,
                    "split": split,
                    "family": "transliteration",
                    "origin": origin or "unknown",
                    "source_text": roman,
                    "reference": native,
                }
            )
        splits[split] = rows
    order = list(range(len(splits["validation"])))
    random.Random(EVAL_SEED).shuffle(order)
    splits[EVAL_SPLIT] = [splits["validation"][i] for i in order[:EVAL_SIZE]]
    return splits, skipped


def write_snapshot(directory, splits, *, source, revision, skipped=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    counts = {}
    for name, rows in splits.items():
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        (directory / f"{name}.jsonl").write_text(text, encoding="utf-8")
        digest.update(name.encode() + b"\0" + text.encode("utf-8"))
        origins = {}
        for r in rows:
            origins[r["origin"]] = origins.get(r["origin"], 0) + 1
        counts[name] = {"tasks": len(rows), "origins": dict(sorted(origins.items()))}
    manifest = {
        "snapshot_id": digest.hexdigest(),
        "source": source,
        "revision": revision,
        "language": "te",
        "families": ["transliteration"],
        "splits": list(splits),
        "eval_splits": [EVAL_SPLIT] if EVAL_SPLIT in splits else [],
        "counts": counts,
        "skipped": skipped or {},
        "license": "CC BY 4.0 (Aksharantar, AI4Bharat)",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare(output, revision=None):
    from huggingface_hub import HfApi, hf_hub_download

    revision = revision or HfApi().dataset_info(REPO).sha
    archive = hf_hub_download(REPO, ARCHIVE, repo_type="dataset", revision=revision)
    splits, skipped = build_tasks(archive)
    return write_snapshot(output, splits, source=f"{REPO}/{ARCHIVE}", revision=revision, skipped=skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("snapshot"))
    parser.add_argument("--revision", default=None, help="dataset commit (default: current main)")
    args = parser.parse_args()
    manifest = prepare(args.output, args.revision)
    print(json.dumps({k: manifest[k] for k in ("snapshot_id", "revision", "counts", "skipped")}, indent=2))


if __name__ == "__main__":
    main()
