#!/usr/bin/env python3
"""
Compare two MLX Whisper ATC models on ATCO2 test set.
"""
import io
import os
import time
import numpy as np
import soundfile as sf
import mlx_whisper
import jiwer
from datasets import load_from_disk

MODELS = {
    "v2 (current)": os.environ.get("ATC_MLX_MODEL_PATH", "models/whisper-atc-mlx"),
    "v3 (jlvdoorn)": os.environ.get("ATC_MLX_MODEL_V3_PATH", "models/whisper-atc-v3-mlx"),
}
N_SAMPLES = 50
SAMPLE_RATE = 16000

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(),
])

def load_samples(n):
    ds = load_from_disk("data/atco2_test")
    ds = ds.select(range(min(n, len(ds))))
    ds = ds.cast_column("audio", ds.features["audio"].__class__(decode=False))
    audio_list, refs = [], []
    for item in ds:
        b = item["audio"].get("bytes")
        p = item["audio"].get("path")
        a, sr = sf.read(io.BytesIO(b)) if b else sf.read(p)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            a = librosa.resample(a, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio_list.append(a.astype(np.float32))
        refs.append(item["text"])
    return audio_list, refs

def run(model_path, audios, refs):
    # warmup
    mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32),
                           path_or_hf_repo=model_path, language="en")
    hyps, lats = [], []
    for i, a in enumerate(audios):
        t0 = time.perf_counter()
        r = mlx_whisper.transcribe(a, path_or_hf_repo=model_path, language="en")
        lats.append(time.perf_counter() - t0)
        hyps.append(r["text"].strip())
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(audios)}")
    wer = jiwer.wer(refs, hyps,
                    reference_transform=transform, hypothesis_transform=transform)
    return hyps, wer, lats

def main():
    print(f"Loading {N_SAMPLES} samples from ATCO2 test set...")
    audios, refs = load_samples(N_SAMPLES)
    print(f"Loaded. Total audio: {sum(len(a)/SAMPLE_RATE for a in audios):.1f}s\n")

    results = {}
    for name, path in MODELS.items():
        print(f"=== {name} ({path}) ===")
        hyps, wer, lats = run(path, audios, refs)
        results[name] = (hyps, wer, lats)
        print(f"  WER: {wer:.2%}")
        print(f"  avg latency: {sum(lats)/len(lats):.2f}s")
        print(f"  avg RTF: {sum(l/(len(a)/SAMPLE_RATE) for l,a in zip(lats,audios))/len(lats):.3f}")
        print()

    # per-sample comparison
    names = list(MODELS.keys())
    print("=" * 80)
    print(f"{'idx':<4}  {'REF / v2 / v3':<}")
    print("-" * 80)
    for i in range(len(refs)):
        w2 = jiwer.wer([refs[i]], [results[names[0]][0][i]],
                      reference_transform=transform, hypothesis_transform=transform)
        w3 = jiwer.wer([refs[i]], [results[names[1]][0][i]],
                      reference_transform=transform, hypothesis_transform=transform)
        mark = "✓" if w3 < w2 else ("=" if w3 == w2 else "✗")
        print(f"[{i+1:03d}] {mark} v2={w2:.0%}  v3={w3:.0%}")
        print(f"      REF: {refs[i]}")
        print(f"      v2 : {results[names[0]][0][i]}")
        print(f"      v3 : {results[names[1]][0][i]}")
        print()

    print("=" * 80)
    print(f"{'Model':<20} {'WER':>8} {'avg lat':>10} {'avg RTF':>10}")
    for name in names:
        _, wer, lats = results[name]
        avg_lat = sum(lats) / len(lats)
        avg_rtf = sum(l/(len(a)/SAMPLE_RATE) for l,a in zip(lats, audios))/len(lats)
        print(f"{name:<20} {wer:>7.2%} {avg_lat:>9.2f}s {avg_rtf:>10.3f}")

if __name__ == "__main__":
    main()
