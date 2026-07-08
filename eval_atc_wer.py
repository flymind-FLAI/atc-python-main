#!/usr/bin/env python3
"""
WER evaluation: whisper-atc fine-tuned model vs. baseline whisper
on ATCO2 1-hour corpus test set (871 samples)
"""

import os
import numpy as np
import whisper
import jiwer
from datasets import load_from_disk

DEVICE = "cpu"  # mps if Metal available
MODEL_PATH = os.environ.get("ATC_HF_MODEL_PATH", "models/whisper-atc-weights")
BASELINE_MODEL = "medium.en"
N_SAMPLES = 50  # start small to get quick results, set None for full 871

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(),
])

def transcribe_samples(model, samples_audio, sample_rate=16000):
    results = []
    for i, audio in enumerate(samples_audio):
        # ensure float32 mono 16kHz
        audio_np = audio.astype(np.float32)
        result = model.transcribe(audio_np, language="en", fp16=False)
        results.append(result["text"].strip())
        if (i + 1) % 10 == 0:
            print(f"  transcribed {i+1}/{len(samples_audio)}")
    return results

def main():
    print("Loading ATCO2 test set...")
    ds = load_from_disk("data/atco2_test")

    if N_SAMPLES:
        ds = ds.select(range(min(N_SAMPLES, len(ds))))
    print(f"Evaluating on {len(ds)} samples")

    # decode audio manually - bypass torchcodec requirement
    import io, soundfile as sf
    ds_raw = ds.cast_column("audio", ds.features["audio"].__class__(decode=False))
    audio_arrays = []
    references = []
    for item in ds_raw:
        audio_bytes = item["audio"].get("bytes")
        audio_path  = item["audio"].get("path")
        if audio_bytes:
            audio_np, sr = sf.read(io.BytesIO(audio_bytes))
        else:
            audio_np, sr = sf.read(audio_path)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != 16000:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        audio_arrays.append(audio_np.astype(np.float32))
        references.append(item["text"])

    # --- Baseline: standard whisper medium.en ---
    print(f"\n[1/2] Loading baseline: {BASELINE_MODEL}")
    baseline = whisper.load_model(BASELINE_MODEL, device=DEVICE)
    print("Transcribing with baseline...")
    hyp_baseline = transcribe_samples(baseline, audio_arrays)
    wer_baseline = jiwer.wer(references, hyp_baseline, reference_transform=transform, hypothesis_transform=transform)
    print(f"Baseline WER: {wer_baseline:.1%}")
    del baseline

    # --- Fine-tuned ATC model (HuggingFace transformers format) ---
    print(f"\n[2/2] Loading fine-tuned ATC model from {MODEL_PATH}")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import torch

    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    atc_hf = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
    atc_hf.eval()

    def transcribe_hf(audio_arrays):
        results = []
        for i, audio in enumerate(audio_arrays):
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                predicted_ids = atc_hf.generate(inputs["input_features"], language="en")
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            results.append(text.strip())
            if (i + 1) % 10 == 0:
                print(f"  transcribed {i+1}/{len(audio_arrays)}")
        return results

    print("Transcribing with ATC model...")
    hyp_atc = transcribe_hf(audio_arrays)
    wer_atc = jiwer.wer(references, hyp_atc, reference_transform=transform, hypothesis_transform=transform)
    print(f"ATC fine-tuned WER: {wer_atc:.1%}")

    # Summary
    print("\n" + "="*40)
    print(f"Samples evaluated : {len(ds)}")
    print(f"Baseline WER      : {wer_baseline:.1%}")
    print(f"ATC fine-tuned WER: {wer_atc:.1%}")
    delta = wer_baseline - wer_atc
    if delta > 0:
        print(f"Improvement       : -{delta:.1%} ✓")
    else:
        print(f"Regression        : +{abs(delta):.1%} ✗  fine-tune made it worse")

    # show all sample comparisons
    print("\nSample comparisons:")
    print("-" * 80)
    for i in range(len(references)):
        w_base = jiwer.wer([references[i]], [hyp_baseline[i]], reference_transform=transform, hypothesis_transform=transform)
        w_atc  = jiwer.wer([references[i]], [hyp_atc[i]],      reference_transform=transform, hypothesis_transform=transform)
        print(f"[{i+1:03d}] base={w_base:.0%}  atc={w_atc:.0%}")
        print(f"  REF : {references[i]}")
        print(f"  BASE: {hyp_baseline[i]}")
        print(f"  ATC : {hyp_atc[i]}")
        print()

if __name__ == "__main__":
    main()
