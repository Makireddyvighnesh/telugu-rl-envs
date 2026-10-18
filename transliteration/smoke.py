"""Check real HTTP/WebSocket transport, independent sessions, no reference leaks, and every reward path.

    python -m telugu_transliteration.smoke                      # synthetic fixture
    python -m telugu_transliteration.smoke --snapshot snapshot  # a prepared snapshot
"""

import argparse
import json
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from .client import connect
from .data.catalog import Catalog
from .fixtures import make_fixture
from .models import TransliterationAction
from .runtime import local_server


def probe(url, catalog, per_split=3):
    started = time.monotonic()
    checked = {}
    with ExitStack() as stack:
        first = stack.enter_context(connect(url))
        second = stack.enter_context(connect(url))
        manifest = first.manifest()
        assert manifest["snapshot_id"] == catalog.snapshot_id
        for split in catalog.splits():
            count = first.num_tasks(split)
            assert count == catalog.count(split), (split, count)
            listed = first.get_task_range(split, 0, min(per_split, count))
            # Discovery must not leak the answer.
            assert all("reference" not in task for task in listed)
            for task in listed:
                task_id, reference = task["task_id"], catalog.get(task["task_id"])["reference"]
                observation = first.reset(task_id=task_id).observation
                repeated = second.reset(task_id=task_id).observation
                assert not observation.done and observation.task_id == repeated.task_id == task_id
                assert "reference" not in observation.model_dump() and not observation.metrics
                assert task["source_text"] in observation.prompt

                exact = first.step(TransliterationAction(answer=reference))
                assert exact.done and exact.reward == 1.0, (task_id, exact.reward)
                assert exact.observation.metrics["exact_match"] is True

                # An empty answer scores 0 and is flagged.
                empty = second.step(TransliterationAction(answer=""))
                assert empty.reward == 0.0 and empty.observation.metrics["empty"] is True

                # One dropped character earns partial credit, below an exact answer.
                second.reset(task_id=task_id)
                partial = second.step(TransliterationAction(answer=reference[:-1]))
                assert 0.0 < partial.reward < 0.8, (task_id, partial.reward)

                # A finished episode can't be graded again.
                try:
                    first.step(TransliterationAction(answer=reference))
                except Exception:
                    pass
                else:
                    raise AssertionError("A second step re-graded a consumed task")
                first.reset(task_id=task_id)
                checked[split] = checked.get(split, 0) + 1
    return {
        "status": "passed",
        "snapshot_id": manifest["snapshot_id"],
        "checked": checked,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with ExitStack() as stack:
        snapshot = args.snapshot
        if snapshot is None:
            snapshot = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="translit-smoke-")))
            make_fixture(snapshot)
        url = stack.enter_context(local_server(snapshot))
        result = probe(url, Catalog(snapshot))
        result["source"] = "prepared-snapshot" if args.snapshot else "synthetic-fixture"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
