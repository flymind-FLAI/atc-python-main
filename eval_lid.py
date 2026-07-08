#!/usr/bin/env python3
"""
Transcribe ATCO2-LID wav samples (no ground truth).
Runs CZEN/EN and EN-AU/EN subsets, outputs results for manual review.
"""
import os, time
import numpy as np
import soundfile as sf
import mlx_whisper

MODEL_PATH = "models/whisper-atc-mlx"
LID_ROOT   = "data/ATCO2-LIDdataset-v1_beta"
SUBSETS    = {
    "CZEN (Prague)":  "LID_EVAL_CZEN/EN",
    "EN-AU (Sydney)": "LID_EVAL_EN-AU/EN",
}
MAX_PER_SUBSET = 100  # set None for all
OUTPUT_FILE    = "lid_eval_result.txt"

def read_info(path):
    info = {}
    try:
        for line in open(path):
            k, v = line.strip().split(":", 1)
            info[k.strip()] = v.strip()
    except Exception:
        pass
    return info

def main():
    # warmup
    print("Warming up model...")
    mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32),
                           path_or_hf_repo=MODEL_PATH, language="en")
    print("Ready.\n")

    lines = []
    summary = []

    for subset_name, rel_path in SUBSETS.items():
        folder = os.path.join(LID_ROOT, rel_path)
        wavs = sorted(f for f in os.listdir(folder) if f.endswith(".wav"))
        if MAX_PER_SUBSET:
            wavs = wavs[:MAX_PER_SUBSET]

        print(f"=== {subset_name} — {len(wavs)} samples ===")
        lines.append(f"\n{'='*70}")
        lines.append(f"{subset_name}  ({folder})")
        lines.append(f"{'='*70}")

        lats, snrs = [], []
        for i, fname in enumerate(wavs):
            base   = fname[:-4]
            wav_p  = os.path.join(folder, fname)
            info_p = os.path.join(folder, base + ".info")

            audio, sr = sf.read(wav_p)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            audio = audio.astype(np.float32)

            info   = read_info(info_p)
            dur    = float(info.get("dur", "0").split()[0])
            snr    = float(info.get("SNR", "0"))

            t0  = time.perf_counter()
            res = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL_PATH, language="en")
            lat = time.perf_counter() - t0
            text = res["text"].strip()

            lats.append(lat)
            snrs.append(snr)

            line = f"[{i+1:04d}] dur={dur:.1f}s SNR={snr:.0f}dB lat={lat:.2f}s\n        {text}"
            lines.append(line)

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(wavs)}")

        avg_lat = sum(lats) / len(lats)
        avg_snr = sum(snrs) / len(snrs)
        avg_rtf = sum(lats) / len(lats) / (sum(
            float(read_info(os.path.join(folder, f[:-4]+".info")).get("dur","4").split()[0])
            for f in wavs) / len(wavs))
        print(f"  avg latency={avg_lat:.2f}s  avg SNR={avg_snr:.1f}dB  RTF={avg_rtf:.3f}\n")
        summary.append(f"{subset_name:<20} samples={len(wavs):4d}  avg_lat={avg_lat:.2f}s  avg_SNR={avg_snr:.1f}dB  RTF={avg_rtf:.3f}")

    lines.append(f"\n{'='*70}")
    lines.append("SUMMARY")
    lines.append(f"{'='*70}")
    lines.extend(summary)

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
