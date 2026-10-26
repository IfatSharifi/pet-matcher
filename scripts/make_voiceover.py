"""Synthesize the Hebrew voiceover for the demo video, one WAV per line.

    python scripts/make_voiceover.py                       # docs/demo_lines_he.json -> build/voice/
    python scripts/record_demo.py --db demo.db --voice build/voice/manifest.json

Voice: Phonikud (https://github.com/thewh1teagle/phonikud) - adds Hebrew vowel marks, converts
to phonemes, and speaks with a Piper model trained on Hebrew. The models are licensed for
non-commercial use (fine for a portfolio video). Setup, one time:

    pip install phonikud-onnx piper-onnx
    pip install --ignore-requires-python git+https://github.com/thewh1teagle/phonikud

The models (~360 MB) download on first run.

The voice adds random variation, so the same line sounds a little different each time and
now and then slurs a word. With --candidates N (default 4) each line is synthesized N times
and a Hebrew speech recognizer (ivrit-ai/whisper-large-v3-turbo, ~1.6 GB) transcribes every
take; the take whose transcript is closest to the script wins. --candidates 1 skips this.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
import wave
from pathlib import Path

import numpy as np

HF = "https://huggingface.co"
MODELS = {  # file -> source
    "phonikud-1.0.int8.onnx": f"{HF}/Phonikud/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx",
    "model.onnx": f"{HF}/Phonikud/phonikud-tts-checkpoints/resolve/main/model.onnx",
    "model.config.json": f"{HF}/Phonikud/phonikud-tts-checkpoints/resolve/main/model.config.json",
}
GAP_S = 0.15  # silence added after each line


def fetch_models(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        if not (folder / name).exists():
            print(f"  downloading {name} ...")
            urllib.request.urlretrieve(url, folder / name)


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    peak = float(np.max(np.abs(samples))) or 1.0
    pcm = (samples / peak * 0.89 * 32767).astype("<i2")  # peak at about -1 dBFS
    pcm = np.concatenate([pcm, np.zeros(int(sr * GAP_S), dtype="<i2")])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def normalize(s: str) -> str:
    s = re.sub(r"[\u0591-\u05C7]", "", s)  # drop vowel marks
    return " ".join(re.sub(r"[^\u05D0-\u05EA0-9a-zA-Z ]+", " ", s).split())


def cer(ref: str, hyp: str) -> float:
    """Character error rate after normalizing punctuation and vowel marks."""
    r, h = normalize(ref), normalize(hyp)
    d = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[len(h)] / max(len(r), 1)


class Judge:
    """Hebrew Whisper: transcribes a take so we can score how clearly it was spoken."""
    def __init__(self, model: str = "ivrit-ai/whisper-large-v3-turbo") -> None:
        from transformers import pipeline
        self.asr = pipeline("automatic-speech-recognition", model=model, device="cpu")

    def transcribe(self, samples: np.ndarray, sr: int) -> str:
        if sr != 16000:
            x = np.linspace(0, len(samples) - 1, int(len(samples) * 16000 / sr))
            samples = np.interp(x, np.arange(len(samples)), samples).astype(np.float32)
        return self.asr(samples, generate_kwargs={"language": "he", "task": "transcribe"})["text"].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="docs/demo_lines_he.json")
    ap.add_argument("--out", default="build/voice")
    ap.add_argument("--models", default="build/tts-models")
    ap.add_argument("--speed", type=float, default=1.0, help=">1 slower, <1 faster (Piper length_scale)")
    ap.add_argument("--candidates", type=int, default=4, help="takes per line; the clearest one wins")
    ap.add_argument("--only", default="", help="comma-separated line ids to redo; keeps the rest")
    args = ap.parse_args()

    from phonikud import phonemize
    from phonikud_onnx import Phonikud
    from piper_onnx import Piper

    models, out = Path(args.models), Path(args.out)
    fetch_models(models)
    out.mkdir(parents=True, exist_ok=True)
    nakdan = Phonikud(str(models / "phonikud-1.0.int8.onnx"))
    voice = Piper(str(models / "model.onnx"), str(models / "model.config.json"))

    judge = Judge() if args.candidates > 1 else None
    only = {x for x in args.only.split(",") if x}
    old = {}
    if only and (out / "manifest.json").exists():
        old = {m["id"]: m for m in json.loads((out / "manifest.json").read_text(encoding="utf-8"))}
    manifest = []
    for line in json.loads(Path(args.lines).read_text(encoding="utf-8")):
        if only and line["id"] not in only and line["id"] in old:
            manifest.append(old[line["id"]])
            continue
        vocalized = nakdan.add_diacritics(line["say"])
        phonemes = phonemize(vocalized)
        best = None
        for _ in range(max(args.candidates, 1)):
            samples, sr = voice.create(phonemes, is_phonemes=True, length_scale=args.speed)
            samples = np.asarray(samples, dtype=np.float32)
            heard, score = "", 0.0
            if judge:
                heard = judge.transcribe(samples, sr)
                # numbers: the recognizer writes digits, the script spells them; accept either form
                score = min(cer(line["say"], heard), cer(line["show"], heard))
            if best is None or score < best[0]:
                best = (score, samples, sr, heard)
            if score == 0:
                break
        score, samples, sr, heard = best
        wav = out / f"{line['id']}.wav"
        write_wav(wav, samples, sr)
        duration = len(samples) / sr + GAP_S
        manifest.append({**line, "wav": wav.name, "seconds": round(duration, 2), "vocalized": vocalized,
                         "heard": heard, "cer": round(score, 3)})
        print(f"  {line['id']:<10} {duration:5.1f}s  CER {score:.3f}  heard: {heard}")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(manifest)} lines, {sum(m['seconds'] for m in manifest):.0f}s of speech -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
