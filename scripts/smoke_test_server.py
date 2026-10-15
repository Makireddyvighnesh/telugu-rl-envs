"""End-to-end smoke test against a running env server.

    python scripts/smoke_test_server.py ws://localhost:8000/ws --episodes 3

For each episode: reset, then replay the same sample_id for a GRPO-style group
of 4 "policies" (perfect / one word dropped / garbage / empty) and print rewards.
"""

import argparse
import asyncio
import json
import time

import websockets


async def call(ws, type_, data):
    await ws.send(json.dumps({"type": type_, "data": data}))
    msg = json.loads(await ws.recv())
    if msg.get("type") == "error":
        raise RuntimeError(msg)
    return msg["data"]


def candidates(ref: str) -> dict[str, str]:
    words = ref.split()
    return {
        "perfect": ref,
        "drop_word": " ".join(words[:-1]) if len(words) > 1 else ref[:-1],
        "garbage": "ఇది తప్పు సమాధానం",
        "empty": "",
    }


async def main(url: str, episodes: int, seed: int):
    async with websockets.connect(url, max_size=2**26) as ws:
        for ep in range(episodes):
            t = time.time()
            obs = (await call(ws, "reset", {"seed": seed}))["observation"]
            sid = obs["sample_id"]
            payload = "image" if obs.get("image_png_b64") else "audio"
            size = len(obs.get("image_png_b64") or obs.get("audio_wav_b64") or "")
            print(f"\n[{ep}] {sid}  {payload} b64={size/1e3:.0f}KB  reset {time.time()-t:.1f}s")
            # The reference is only revealed after a step; probe once, then run the group.
            ref = (await call(ws, "step", {"transcription": ""}))["observation"]["reference"]
            print(f"    REF: {ref[:100]!r}")
            for name, text in candidates(ref).items():
                await call(ws, "reset", {"sample_id": sid})
                res = await call(ws, "step", {"transcription": text})
                o = res["observation"]
                print(f"    {name:10s} reward={res['reward']:.3f} cer={o['cer']:.3f} wer={o['wer']:.3f} flags={o['flags']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    asyncio.run(main(a.url, a.episodes, a.seed))
