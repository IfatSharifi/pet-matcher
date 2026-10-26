# Demo video

`docs/demo.mp4`: 84 seconds, phone format (780x1688), Hebrew captions and a Hebrew voiceover, recorded from the real app.

The words live in [`demo_lines_he.json`](demo_lines_he.json), one entry per screen: `show` is the caption, `say` is what the voice reads (numbers spelled out, "Claude" written as קלוד). Captions match the speech.

## How it's made

1. **Voice** - `scripts/make_voiceover.py` synthesizes each line with [Phonikud](https://github.com/thewh1teagle/phonikud), an open Hebrew text-to-speech project. It adds vowel marks, converts them to phonemes, and speaks with a Piper voice trained on Hebrew.
2. **Quality gate** - the voice varies a little on every run and occasionally slurs a word. So each line is synthesized up to 4 times, a Hebrew speech recognizer ([ivrit-ai/whisper-large-v3-turbo](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo)) transcribes every take, and the take closest to the script wins. Lines the voice couldn't say cleanly were reworded: it can't pronounce "WhatsApp" in any spelling, so the intro says "ברשתות החברתיות". Final result: 11 of 14 lines transcribe perfectly, and the other 3 differ only in how numbers are written.
3. **Recording** - `scripts/record_demo.py` drives the app in Chrome. Each line shows its caption and starts its audio; the on-screen actions run while it speaks; the next line waits for it to finish, plus a 0.35-second breath. The audio clips are then placed at the exact moments their captions appeared (`demo.cues.json`).

## Re-recording

```powershell
pip install phonikud-onnx piper-onnx playwright
pip install --ignore-requires-python git+https://github.com/thewh1teagle/phonikud
playwright install chromium                                  # and install ffmpeg
python scripts\make_voiceover.py                              # -> build\voice\ (~7 min with the quality gate)
python scripts\seed_demo.py --dogs 300 --db demo.db
$env:PETMATCH_DB = "demo.db"; uvicorn app:app                # one window, with ANTHROPIC_API_KEY set
python scripts\record_demo.py --db demo.db --voice build\voice\manifest.json   # another window
```

Change a line: edit `demo_lines_he.json`, then `python scripts\make_voiceover.py --only <id>` regenerates just that line.

## Licenses

The Phonikud voice models are licensed for non-commercial use only, which covers a portfolio video. A commercial video would need a different voice, or your own.
