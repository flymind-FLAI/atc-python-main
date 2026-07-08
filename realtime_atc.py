#!/usr/bin/env python3
"""
Real-time ATC transcription simulation — ATCO2 dataset, no mic needed.
Shows REF vs ATC model output side by side as if streaming.
Press Ctrl+C to stop.
"""

import numpy as np
import time
import io
import os
import soundfile as sf
from datasets import load_from_disk
import mlx_whisper
import jiwer

MODEL_PATH  = os.environ.get("ATC_MLX_MODEL_PATH", "models/whisper-atc-mlx")
SAMPLE_RATE = 16000
PAUSE_SEC   = 1.5   # pause between samples to simulate real-time


transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(),
])

def load_model():
    print("Loading ATC model (MLX, Neural Engine)...")
    # warmup
    import numpy as np
    dummy = np.zeros(16000, dtype=np.float32)
    mlx_whisper.transcribe(dummy, path_or_hf_repo=MODEL_PATH, language="en")
    print("Model ready.\n")

def transcribe(audio_np):
    result = mlx_whisper.transcribe(audio_np, path_or_hf_repo=MODEL_PATH, language="en")
    text = result["text"].strip()
    words = text.split()
    if len(words) > 4 and len(set(words)) / len(words) < 0.3:
        return "[repetition filtered]"
    return text

def main():
    load_model()

    print("Loading ATCO2 dataset...")
    ds = load_from_disk("data/atco2_test")
    ds_raw = ds.cast_column("audio", ds.features["audio"].__class__(decode=False))
    print(f"Dataset: {len(ds_raw)} samples\n")

    print("=" * 70)
    print(f"{'LIVE ATC TRANSCRIPTION':^70}")
    print("=" * 70)
    print()

    total_wer = []
    latencies = []

    try:
        for i, item in enumerate(ds_raw):
            # load audio
            audio_bytes = item["audio"].get("bytes")
            audio_path  = item["audio"].get("path")
            if audio_bytes:
                audio_np, sr = sf.read(io.BytesIO(audio_bytes))
            else:
                audio_np, sr = sf.read(audio_path)

            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=SAMPLE_RATE)

            duration = len(audio_np) / SAMPLE_RATE
            ref = item["text"]

            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] #{i+1:03d}  ({duration:.1f}s)  transcribing...", end="\r")

            t0 = time.perf_counter()
            hyp = transcribe(audio_np.astype(np.float32))
            latency = time.perf_counter() - t0

            w = jiwer.wer([ref], [hyp], reference_transform=transform, hypothesis_transform=transform)
            total_wer.append(w)
            latencies.append(latency)

            ts = time.strftime("%H:%M:%S")
            rtf = latency / duration  # real-time factor
            print(f"[{ts}] #{i+1:03d}  audio={duration:.1f}s  latency={latency:.2f}s  RTF={rtf:.2f}  WER={w:.0%}")
            print(f"  REF: {ref}")
            print(f"  ATC: {hyp}")
            print()

            time.sleep(PAUSE_SEC)

    except KeyboardInterrupt:
        pass

    if total_wer:
        avg_latency = sum(latencies) / len(latencies)
        avg_rtf = sum(l / d for l, d in zip(latencies, [len(ds_raw[i]["text"]) for i in range(len(latencies))])) if latencies else 0
        print("=" * 70)
        print(f"Samples processed : {len(total_wer)}")
        print(f"Average WER       : {sum(total_wer)/len(total_wer):.1%}")
        print(f"Avg latency       : {avg_latency:.2f}s")
        print(f"Min latency       : {min(latencies):.2f}s")
        print(f"Max latency       : {max(latencies):.2f}s")

if __name__ == "__main__":
    main()
