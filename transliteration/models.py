from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TransliterationAction(Action):
    answer: str = Field(default="", max_length=4096)


class TransliterationObservation(Observation):
    task_id: str = ""
    snapshot_id: str = ""
    split: str = ""
    family: str = ""
    origin: str = ""
    source_text: str = ""
    prompt: str = ""
    # Filled by step(): cer, wer, exact_match, empty, degenerate. Never the reference.
    metrics: dict[str, float | bool] = Field(default_factory=dict)
    grading_policy_id: str = ""
