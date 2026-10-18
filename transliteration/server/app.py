import os

from openenv.core.env_server import create_app

from ..models import TransliterationAction, TransliterationObservation
from .environment import TransliterationEnvironment, configured_catalog
from .gradio_ui import build_ui
from .rewards import ERROR_WEIGHT, EXACT_WEIGHT, GRADING_POLICY


def create_server():
    catalog = configured_catalog()
    os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
    app = create_app(
        lambda: TransliterationEnvironment(catalog),
        TransliterationAction,
        TransliterationObservation,
        env_name="telugu_transliteration",
        max_concurrent_envs=int(os.environ.get("TRANSLIT_MAX_SESSIONS", "64")),
        gradio_builder=build_ui,
        custom_tab_name="Try it",
        custom_tab_primary=True,
        show_default_tab=False,
        title_override="Telugu transliteration (Aksharantar)",
    )

    @app.get("/healthz")
    def health():
        return {"status": "ok", "snapshot_id": catalog.snapshot_id}

    @app.get("/manifest")
    def manifest():
        return {
            **catalog.manifest,
            "grading": {
                "policy": GRADING_POLICY,
                "reward": f"{ERROR_WEIGHT} * (1 - cer) + {EXACT_WEIGHT} * exact_match; 0 if empty or degenerate",
                "cer": "Levenshtein over Unicode code points after normalization, divided by reference length, capped at 1",
                "normalization": "NFKC/NFC, zero-width and quote removal, danda/dash folding, Telugu digits to ASCII",
            },
        }

    return app


def main():
    import uvicorn

    uvicorn.run(
        "telugu_transliteration.server.app:create_server",
        factory=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        ws="websockets",
    )


if __name__ == "__main__":
    main()
