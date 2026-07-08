#!/usr/bin/env python3
"""
ATC Transcribe — macOS Desktop App
Streaming transcription + real-time speaker diarization (CTR / PLT)
"""

import customtkinter as ctk
import tkinter as tk
import sounddevice as sd
import numpy as np
import mlx_whisper
import threading
import queue
import time
import traceback
import sys
import os
from resemblyzer import VoiceEncoder, preprocess_wav

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH      = os.environ.get("ATC_MLX_MODEL_PATH", "models/whisper-atc-mlx")
SAMPLE_RATE     = 16000
BLOCK_MS        = 80
BLOCK_SIZE      = int(SAMPLE_RATE * BLOCK_MS / 1000)
INFERENCE_EVERY = 0.5
MAX_BUFFER_SEC  = 60.0   # safety cap only; normal boundary is silence-based commit
SILENCE_THRESH  = 0.005
# ─────────────────────────────────────────────────────────────────────────────

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SPEAKER_COLORS = {
    "ATC": "#4a9eff",   # blue — controller
    "PLT": "#4caf50",   # green — pilot
    "?":   "#aaaaaa",   # grey — unknown
}


def get_input_devices():
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0]


def rms(block):
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


ICAO = {
    "alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel",
    "india","juliet","kilo","lima","mike","november","oscar","papa",
    "quebec","romeo","sierra","tango","uniform","victor","whiskey",
    "xray","yankee","zulu",
}
ATC_CAPS = {
    "ils","rnp","qnh","atc","ifr","vfr","ctaf","ils","atis",
}
ATC_TITLE = {
    "ruzyne","praha","radar","tower","approach","baltu","vlasim",
    "lanux","benesov","liege","wien","warsaw","speedbird","ryanair",
    "lufthansa","eurowings","csa","belavia","skytravel","klm",
}

DIGIT_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,
    "five":5,"six":6,"seven":7,"eight":8,"nine":9,"niner":9,
}
MULTIPLIERS  = {"hundred": 100, "thousand": 1000}
DECIMAL_WORDS = {"decimal", "point"}

def normalize_case(text: str) -> str:
    words = text.split()
    out = []
    for w in words:
        lw = w.lower().rstrip(".,;:")
        punct = w[len(lw):]
        if lw in ICAO or lw in ATC_TITLE:
            out.append(lw.capitalize() + punct)
        elif lw in ATC_CAPS:
            out.append(lw.upper() + punct)
        else:
            out.append(w)
    return " ".join(out)

def _span_to_numstr(lows):
    """Convert a list of lowercased span tokens to a numeric string."""
    if any(t in DECIMAL_WORDS for t in lows):
        dec_i = next(i for i, t in enumerate(lows) if t in DECIMAL_WORDS)
        int_part = "".join(str(DIGIT_WORDS[t]) for t in lows[:dec_i] if t in DIGIT_WORDS) or "0"
        frac_part = "".join(str(DIGIT_WORDS[t]) for t in lows[dec_i+1:] if t in DIGIT_WORDS)
        return int_part + "." + frac_part
    if any(t in MULTIPLIERS for t in lows):
        val = cur = 0
        for t in lows:
            if t in DIGIT_WORDS:
                cur = cur * 10 + DIGIT_WORDS[t]
            elif t == "thousand":
                val += cur * 1000; cur = 0
            elif t == "hundred":
                val += cur * 100; cur = 0
        return str(val + cur)
    return "".join(str(DIGIT_WORDS[t]) for t in lows if t in DIGIT_WORDS)

def normalize_numbers(text: str) -> str:
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        w = words[i]
        core = w.lower().rstrip(".,;:")
        trail = w[len(core):]

        is_digit_start   = core in DIGIT_WORDS and not trail
        is_decimal_start = (core in DECIMAL_WORDS and not trail and
                            i + 1 < len(words) and
                            words[i+1].lower().rstrip(".,;:") in DIGIT_WORDS)

        if is_digit_start or is_decimal_start:
            span_orig = [w]
            span_lows = [core]
            last_trail = trail
            i += 1

            while i < len(words) and not last_trail:
                nw = words[i]
                nc = nw.lower().rstrip(".,;:")
                np_ = nw[len(nc):]

                if nc in DIGIT_WORDS or nc in MULTIPLIERS:
                    span_orig.append(nw)
                    span_lows.append(nc)
                    last_trail = np_
                    i += 1
                elif nc in DECIMAL_WORDS and not np_:
                    if i + 1 < len(words) and words[i+1].lower().rstrip(".,;:") in DIGIT_WORDS:
                        span_orig.append(nw)
                        span_lows.append(nc)
                        i += 1
                    else:
                        break
                else:
                    break

            # only annotate multi-token spans or spans with decimal/multiplier
            has_special = any(t in DECIMAL_WORDS or t in MULTIPLIERS for t in span_lows)
            if len(span_orig) >= 2 or has_special:
                numstr = _span_to_numstr(span_lows)
                result.append(" ".join(span_orig) + f" ({numstr})" + last_trail)
            else:
                result.append(" ".join(span_orig) + last_trail)
        else:
            result.append(w)
            i += 1
    return " ".join(result)


def _is_hallucination(text: str) -> bool:
    words = text.split()
    # repeated-word loop: "the the the..."
    if len(words) > 4 and len(set(w.lower() for w in words)) / len(words) < 0.35:
        return True
    # glued-repeat loop: "unityunityunity..." — any single token absurdly long
    if any(len(w) > 25 for w in words):
        return True
    # short-substring repeating inside one token (covers cases without spaces)
    for w in text.split():
        lw = w.lower()
        if len(lw) >= 12:
            for k in range(2, 7):
                if len(lw) >= k * 4 and lw[:k] * (len(lw) // k) == lw[:k * (len(lw) // k)]:
                    return True
    return False


def stable_prefix(prev: str, curr: str) -> str:
    pw, cw = prev.split(), curr.split()
    i = 0
    for i, (a, b) in enumerate(zip(pw, cw)):
        if a.lower() != b.lower():
            return " ".join(cw[:i])
    return " ".join(cw[:min(len(pw), len(cw))])


class SpeakerTracker:
    """
    Maintains embeddings for up to 2 speakers (CTR / PLT).
    First segment = Speaker A, second distinct voice = Speaker B.
    Labels assigned by order of first appearance.
    """
    def __init__(self, encoder: VoiceEncoder, threshold=0.82):
        self.encoder   = encoder
        self.threshold = threshold
        self.speakers  = {}    # label -> mean embedding
        self.labels    = ["ATC", "PLT"]
        self.next_idx  = 0

    def identify(self, audio: np.ndarray) -> str:
        try:
            wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
            emb = self.encoder.embed_utterance(wav)
        except Exception:
            return "?"

        best_label, best_sim = None, -1
        for label, mean_emb in self.speakers.items():
            sim = float(np.dot(emb, mean_emb) /
                        (np.linalg.norm(emb) * np.linalg.norm(mean_emb) + 1e-8))
            if sim > best_sim:
                best_sim, best_label = sim, label

        if best_sim >= self.threshold:
            # update running mean
            n = self.speakers.get(f"_n_{best_label}", 1)
            self.speakers[best_label] = (self.speakers[best_label] * n + emb) / (n + 1)
            self.speakers[f"_n_{best_label}"] = n + 1
            return best_label
        else:
            if self.next_idx < len(self.labels):
                label = self.labels[self.next_idx]
                self.next_idx += 1
                self.speakers[label] = emb
                self.speakers[f"_n_{label}"] = 1
                return label
            return "?"

    def reset(self):
        self.speakers  = {}
        self.next_idx  = 0


class ATCApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ATC Transcribe")
        self.geometry("900x680")
        self.resizable(True, True)

        self.audio_q      = queue.Queue()
        self.stream       = None
        self.running      = False
        self.model_ready  = False
        self.encoder      = None
        self.tracker      = None
        self.tracker_lock = threading.Lock()
        self.silence_cutoff = 0.6   # plain float, read by bg thread

        # streaming state
        self.audio_blocks = []
        self.last_text    = ""
        self.confirmed    = ""
        self.silence_secs = 0.0
        self.last_infer_t = 0.0
        self.is_speaking  = False
        self.line_count   = 0
        self._live_idx    = None   # "line.col" string of where live text starts

        self._build_ui()
        self._load_models()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = ctk.CTkFrame(self, height=56, corner_radius=0)
        top.pack(fill="x")
        top.pack_propagate(False)
        ctk.CTkLabel(top, text="ATC Transcribe",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=16, pady=10)
        self.status_lbl = ctk.CTkLabel(top, text="⏳ Loading...",
                                        font=ctk.CTkFont(size=13), text_color="#aaa")
        self.status_lbl.pack(side="left", padx=8)

        # Device row
        dev = ctk.CTkFrame(self, corner_radius=8)
        dev.pack(fill="x", padx=16, pady=(10, 0))
        ctk.CTkLabel(dev, text="Input:", font=ctk.CTkFont(size=13)).pack(side="left", padx=12, pady=8)
        self.devices   = get_input_devices()
        self.dev_names = [f"[{i}] {n}" for i, n in self.devices]
        self.dev_var   = ctk.StringVar(value=self.dev_names[0] if self.dev_names else "—")
        ctk.CTkOptionMenu(dev, variable=self.dev_var, values=self.dev_names,
                           width=280).pack(side="left", padx=6, pady=8)
        ctk.CTkButton(dev, text="↻", width=36,
                       command=self._refresh_devs).pack(side="left", padx=2)

        # Settings row
        sett = ctk.CTkFrame(self, corner_radius=8)
        sett.pack(fill="x", padx=16, pady=(8, 0))

        ctk.CTkLabel(sett, text="Silence cutoff:",
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=12, pady=6)
        self.silence_var = ctk.DoubleVar(value=0.6)
        self.silence_slider = ctk.CTkSlider(sett, from_=0.2, to=2.0, number_of_steps=18,
                                             variable=self.silence_var, width=160,
                                             command=self._on_silence_change)
        self.silence_slider.pack(side="left", padx=6, pady=6)
        self.silence_val_lbl = ctk.CTkLabel(sett, text="0.6s",
                                             font=ctk.CTkFont(size=12), width=36)
        self.silence_val_lbl.pack(side="left")

        # Speaker reset button
        ctk.CTkButton(sett, text="Reset Speakers", width=130, height=28,
                       fg_color="#444", hover_color="#555",
                       command=self._reset_speakers).pack(side="right", padx=12)

        # Legend
        for label, color in [("● ATC", "#4a9eff"), ("● PLT", "#4caf50")]:
            ctk.CTkLabel(sett, text=label, font=ctk.CTkFont(size=12),
                         text_color=color).pack(side="right", padx=8)

        # Metrics
        met = ctk.CTkFrame(self, corner_radius=8)
        met.pack(fill="x", padx=16, pady=(8, 0))
        self.m_lat  = self._metric(met, "Latency", "—")
        self.m_rtf  = self._metric(met, "RTF", "—")
        self.m_segs = self._metric(met, "Lines", "0")
        self.vad_lbl = ctk.CTkLabel(met, text="●  Silence",
                                     font=ctk.CTkFont(size=13), text_color="#666")
        self.vad_lbl.pack(side="right", padx=16)

        # Bottom bar — must pack BEFORE the expanding transcript frame
        bot = ctk.CTkFrame(self, height=56, corner_radius=0)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)
        self.start_btn = ctk.CTkButton(bot, text="▶  Start", width=120, height=36,
                                        font=ctk.CTkFont(size=14, weight="bold"),
                                        fg_color="#2d7a2d", hover_color="#3a9e3a",
                                        command=self._toggle, state="disabled")
        self.start_btn.pack(side="left", padx=16, pady=10)
        ctk.CTkButton(bot, text="Clear", width=72, height=36,
                       fg_color="#444", hover_color="#555",
                       command=self._clear).pack(side="left", padx=4, pady=10)

        # Live line — above history, auto-grows when wrapped text exceeds one row
        ctk.CTkLabel(self, text="Live",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=16, pady=(10, 2))
        self.live_txt = tk.Text(self, bg="#242431", fg="#ffffff",
                                font=("Menlo", 13), wrap="word",
                                relief="flat", bd=0, padx=8, pady=6,
                                height=1, state="disabled", cursor="arrow",
                                highlightthickness=0)
        self.live_txt.pack(fill="x", padx=16, pady=(0, 6))
        self.live_txt.tag_config("conf", foreground="#ffffff")
        self.live_txt.tag_config("tent", foreground="#555555")

        # Transcript history (committed lines only) — fills remaining space
        ctk.CTkLabel(self, text="Transcription",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=16, pady=(6, 2))
        txt_frame = ctk.CTkFrame(self, corner_radius=8)
        txt_frame.pack(fill="both", expand=True, padx=16, pady=(0, 4))

        self.txt_scroll = ctk.CTkScrollbar(txt_frame, orientation="vertical",
                                            width=12,
                                            fg_color="#242431",
                                            button_color="#444",
                                            button_hover_color="#555")
        self.txt_scroll.pack(side="right", fill="y")
        self.txt = tk.Text(txt_frame, bg="#242431", fg="#e6e6e6",
                           font=("Menlo", 13), wrap="word",
                           relief="flat", bd=0, padx=12, pady=10,
                           state="disabled", cursor="arrow",
                           highlightthickness=0,
                           yscrollcommand=self.txt_scroll.set)
        self.txt_scroll.configure(command=self.txt.yview)
        self.txt.pack(side="left", fill="both", expand=True)

        self.txt.tag_config("ts",  foreground="#888888", font=("Menlo", 11))
        self.txt.tag_config("ATC", foreground="#4a9eff", font=("Menlo", 13, "bold"))
        self.txt.tag_config("PLT", foreground="#4caf50", font=("Menlo", 13, "bold"))
        self.txt.tag_config("?",   foreground="#aaaaaa", font=("Menlo", 13, "bold"))

    def _metric(self, parent, label, value):
        f = ctk.CTkFrame(parent, corner_radius=6)
        f.pack(side="left", padx=8, pady=6)
        ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=11), text_color="#888").pack(padx=10, pady=(4, 0))
        v = ctk.CTkLabel(f, text=value, font=ctk.CTkFont(size=16, weight="bold"))
        v.pack(padx=10, pady=(0, 4))
        return v

    def _on_silence_change(self, val):
        self.silence_cutoff = float(val)
        self.silence_val_lbl.configure(text=f"{float(val):.1f}s")

    # ── Models ───────────────────────────────────────────────────────────────

    def _load_models(self):
        def _load():
            # warmup whisper
            mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32),
                                   path_or_hf_repo=MODEL_PATH, language="en")
            # load speaker encoder
            self.encoder = VoiceEncoder()
            self.tracker = SpeakerTracker(self.encoder)
            self.model_ready = True
            self.after(0, lambda: (
                self.status_lbl.configure(text="✓ Ready", text_color="#4caf50"),
                self.start_btn.configure(state="normal")
            ))
        threading.Thread(target=_load, daemon=True).start()

    # ── Audio ────────────────────────────────────────────────────────────────

    def _audio_cb(self, indata, frames, t, status):
        self.audio_q.put(indata[:, 0].copy())

    def _selected_dev(self):
        sel = self.dev_var.get()
        for idx, name in self.devices:
            if f"[{idx}] {name}" == sel:
                return idx
        return None

    def _start_stream(self):
        self.stream = sd.InputStream(device=self._selected_dev(),
                                      samplerate=SAMPLE_RATE, channels=1,
                                      blocksize=BLOCK_SIZE, dtype="float32",
                                      callback=self._audio_cb)
        self.stream.start()

    def _stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    # ── Streaming loop ───────────────────────────────────────────────────────

    def _stream_loop(self):
        try:
            self._stream_loop_inner()
        except Exception:
            print("=== stream loop crashed ===", file=sys.stderr)
            traceback.print_exc()
            self.running = False
            self.after(0, lambda: self.status_lbl.configure(
                text="✗ Error (see console)", text_color="#e53935"))

    def _stream_loop_inner(self):
        max_blocks = int(MAX_BUFFER_SEC * 1000 / BLOCK_MS)
        self.last_infer_t = time.perf_counter()

        while self.running:
            try:
                block = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            level = rms(block)
            speaking = level > SILENCE_THRESH
            commit_silence = self.silence_cutoff

            # Only accumulate audio while speech is (or just was) active — prevents
            # inter-utterance silence from filling the buffer and truncating the
            # start of the next long utterance.
            if speaking or self.is_speaking:
                self.audio_blocks.append(block)
                if len(self.audio_blocks) > max_blocks:
                    self.audio_blocks = self.audio_blocks[-max_blocks:]

            if speaking:
                self.silence_secs = 0.0
                if not self.is_speaking:
                    self.is_speaking = True
                    self.after(0, lambda: self.vad_lbl.configure(
                        text="●  Speech", text_color="#4caf50"))
            else:
                if self.is_speaking:
                    self.silence_secs += BLOCK_MS / 1000
                    if self.silence_secs >= commit_silence:
                        text = (self.last_text or self.confirmed).strip()
                        utterance_audio = (np.concatenate(self.audio_blocks).astype(np.float32)
                                           if self.audio_blocks else None)
                        self.audio_blocks = []
                        self.confirmed    = ""
                        self.last_text    = ""
                        self.silence_secs = 0.0
                        self.is_speaking  = False
                        self.after(0, lambda: self.vad_lbl.configure(
                            text="●  Silence", text_color="#666"))
                        if text and not _is_hallucination(text):
                            ts = time.strftime("%H:%M:%S")
                            self.line_count += 1
                            n = self.line_count
                            threading.Thread(
                                target=self._identify_and_commit,
                                args=(ts, normalize_numbers(normalize_case(text)), n, utterance_audio),
                                daemon=True).start()
                        continue

            # only infer while speech is detected
            if not self.is_speaking:
                continue

            # inference every INFERENCE_EVERY seconds
            now = time.perf_counter()
            if now - self.last_infer_t < INFERENCE_EVERY or len(self.audio_blocks) < 4:
                continue

            self.last_infer_t = now
            audio = np.concatenate(self.audio_blocks).astype(np.float32)
            duration = len(audio) / SAMPLE_RATE

            t0 = time.perf_counter()
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL_PATH, language="en")
            latency = time.perf_counter() - t0
            curr = result["text"].strip()

            # Drain backlog: inference blocks the loop; any audio that arrived
            # during inference would otherwise pile up and we'd miss silence
            # between utterances. Process VAD on drained blocks only (no infer).
            while True:
                try:
                    extra = self.audio_q.get_nowait()
                except queue.Empty:
                    break
                extra_level = rms(extra)
                extra_speaking = extra_level > SILENCE_THRESH
                if extra_speaking or self.is_speaking:
                    self.audio_blocks.append(extra)
                    if len(self.audio_blocks) > max_blocks:
                        self.audio_blocks = self.audio_blocks[-max_blocks:]
                if extra_speaking:
                    self.silence_secs = 0.0
                elif self.is_speaking:
                    self.silence_secs += BLOCK_MS / 1000

            stable = stable_prefix(self.last_text, curr)
            if len(stable) > len(self.confirmed):
                self.confirmed = stable

            tentative = curr[len(self.confirmed):].strip()
            self.last_text = curr
            rtf = latency / duration if duration > 0 else 0

            self.after(0, lambda c=self.confirmed, ten=tentative, lat=latency, r=rtf:
                       self._update_live(c, ten, lat, r))

    # ── Display ──────────────────────────────────────────────────────────────

    def _set_live(self, confirmed, tentative):
        self.live_txt.configure(state="normal")
        self.live_txt.delete("1.0", "end")
        if confirmed:
            self.live_txt.insert("end", confirmed, "conf")
        if tentative:
            self.live_txt.insert("end", " " + tentative, "tent")
        # auto-grow: count wrapped display lines, clamp to [1, 6]
        self.live_txt.update_idletasks()
        try:
            last = self.live_txt.index("end-1c")
            disp_line = int(self.live_txt.count("1.0", last, "displaylines")[0]) + 1
        except Exception:
            disp_line = 1
        new_h = max(1, min(6, disp_line))
        if int(self.live_txt.cget("height")) != new_h:
            self.live_txt.configure(height=new_h)
        self.live_txt.configure(state="disabled")

    def _update_live(self, confirmed, tentative, latency, rtf):
        self._set_live(confirmed, tentative)
        self.m_lat.configure(text=f"{latency:.2f}s")
        self.m_rtf.configure(text=f"{rtf:.2f}")

    def _identify_and_commit(self, ts, text, n, audio):
        speaker = "?"
        if audio is not None and self.tracker is not None and len(audio) > SAMPLE_RATE // 2:
            with self.tracker_lock:
                try:
                    speaker = self.tracker.identify(audio)
                except Exception:
                    speaker = "?"
        self.after(0, lambda: self._commit_line(ts, speaker, text, n))

    def _commit_line(self, ts, speaker, text, count):
        self._set_live("", "")
        self.txt.configure(state="normal")
        # newest on top: insert at the very beginning
        self.txt.insert("1.0", text + "\n", ())
        self.txt.insert("1.0", f"{speaker}: ", speaker)
        self.txt.insert("1.0", f"[{ts}] ", "ts")
        self.txt.see("1.0")
        self.txt.configure(state="disabled")
        self.m_segs.configure(text=str(count))

    # ── Controls ─────────────────────────────────────────────────────────────

    def _toggle(self):
        if not self.running:
            self.running      = True
            self.audio_blocks = []
            self.last_text    = ""
            self.confirmed    = ""
            self.silence_secs = 0.0
            self.is_speaking  = False
            self._start_stream()
            threading.Thread(target=self._stream_loop, daemon=True).start()
            self.start_btn.configure(text="■  Stop",
                                      fg_color="#8b1a1a", hover_color="#b02020")
            self.status_lbl.configure(text="🎙 Listening...", text_color="#4caf50")
        else:
            self.running = False
            self._stop_stream()
            self.start_btn.configure(text="▶  Start",
                                      fg_color="#2d7a2d", hover_color="#3a9e3a")
            self.status_lbl.configure(text="✓ Ready", text_color="#4caf50")
            self.vad_lbl.configure(text="●  Silence", text_color="#666")

    def _clear(self):
        self._set_live("", "")
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.configure(state="disabled")
        for m in [self.m_lat, self.m_rtf]:
            m.configure(text="—")
        self.m_segs.configure(text="0")
        self.line_count = 0

    def _reset_speakers(self):
        if self.tracker:
            self.tracker.reset()
        self.txt.configure(state="normal")
        self.txt.insert("end", "── Speakers reset ──\n", "ts")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _refresh_devs(self):
        self.devices   = get_input_devices()
        self.dev_names = [f"[{i}] {n}" for i, n in self.devices]

    def on_close(self):
        self.running = False
        self._stop_stream()
        self.destroy()


if __name__ == "__main__":
    app = ATCApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
