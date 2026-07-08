#!/usr/bin/env python3
"""
ATCO2 Dataset Player — plays samples through system audio output
Routes through Multi-Output Device → BlackHole + speakers simultaneously
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import io
import time
from datasets import load_from_disk

SAMPLE_RATE = 16000
PAUSE_SEC   = 2.0   # pause between samples

def main():
    print("Loading ATCO2 dataset...")
    ds = load_from_disk("data/atco2_test")
    ds_raw = ds.cast_column("audio", ds.features["audio"].__class__(decode=False))
    print(f"{len(ds_raw)} samples loaded\n")
    print("Playing through system output (BlackHole + speakers)...")
    print("Make sure ATC Transcribe app is running and set to BlackHole 2ch\n")
    print("Press Ctrl+C to stop.\n")
    print("─" * 50)

    for i, item in enumerate(ds_raw):
        audio_bytes = item["audio"].get("bytes")
        audio_path  = item["audio"].get("path")
        audio_np, sr = sf.read(io.BytesIO(audio_bytes)) if audio_bytes else sf.read(audio_path)

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=SAMPLE_RATE)

        duration = len(audio_np) / SAMPLE_RATE
        print(f"[{i+1:03d}] ({duration:.1f}s)  {item['text']}")

        sd.play(audio_np.astype(np.float32), samplerate=SAMPLE_RATE)
        sd.wait()
        time.sleep(PAUSE_SEC)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sd.stop()
        print("\nStopped.")
