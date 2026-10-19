"""
Lecture Helper - fully local, private, no API keys.

Listens to whatever your speakers play (Zoom / Teams / browser) through WASAPI
loopback (no microphone), cuts the audio into utterances at pauses, transcribes
each one with faster-whisper, and when the professor asks a question or says
your name it asks a local Ollama model for a 1-2 sentence answer.
Press "Answer now" (or Enter) at any time to force an answer on the last thing heard.

Nothing is written to disk: audio and transcript live in RAM only and the
transcript window is pruned to the last TRANSCRIPT_WINDOW_SEC seconds.

Run:    python lecture_helper.py
Check:  python lecture_helper.py --check    (lists output devices, tests whisper + ollama, then exits)
Flags:  --device NAME   start listening to the output device whose name contains NAME
        --visible       do not hide the window from screen capture

Which audio is heard: WASAPI loopback captures one OUTPUT DEVICE, not one app.
Pick the device in the "Listen to" dropdown. To hear only Zoom/Teams, give that app
its own output device in Windows: Settings > System > Sound > Volume mixer > (app) >
Output device, then select that device here. A second headset, monitor speakers or a
virtual cable (VB-Cable) all work as that separate device.
"""

import collections
import ctypes
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

# ------------------------------------------------------------------ CONFIG ---
MY_NAMES = []                      # e.g. ["Ziad", "Zizo"]  -> any of these triggers an answer
WHISPER_MODEL = "small.en"         # "base.en" = faster, "small.en" = better with accents
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:4b"
NOTES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes.txt")
MAX_NOTES_CHARS = 12000            # keep prompt inside num_ctx
TRANSCRIPT_WINDOW_SEC = 60
AUTO_ANSWER = True                 # answer automatically on questions / your name
ANSWER_COOLDOWN_SEC = 6            # don't auto-answer more often than this
HIDE_FROM_SCREEN_SHARE = True      # window invisible to Zoom/Teams screen capture (Win10 2004+)

# voice activity (energy based)
TARGET_SR = 16000
FRAME_SEC = 0.03
MIN_SILENCE_SEC = 0.8              # pause length that ends an utterance
MIN_UTTERANCE_SEC = 0.6            # shorter blips are ignored
MAX_UTTERANCE_SEC = 15             # force a cut on long monologues
PRE_ROLL_SEC = 0.3
ABS_SPEECH_RMS = 0.004
NOISE_MULT = 3.0

QUESTION_WORDS = (
    "what", "why", "how", "when", "where", "who", "whom", "which", "whose",
    "define", "explain", "describe", "compare", "list", "name", "state",
    "calculate", "derive", "prove", "give me", "tell me", "can anyone", "can someone",
    "anyone", "anybody", "does anyone", "do you", "do we", "did", "is it", "is this",
    "is that", "are they", "are these", "could you", "would you", "should", "what's",
    "whats", "who's", "whos",
)
HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching.", "thanks for watching",
    "you", "bye.", "bye", "thank you very much.", ".", "the end.",
}
# -----------------------------------------------------------------------------


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------- GPU runtime ---
def _add_nvidia_dlls():
    """Make pip-installed cuBLAS / cuDNN (nvidia-*-cu12 wheels) visible to ctranslate2."""
    try:
        import nvidia  # namespace package from the nvidia-*-cu12 wheels
    except ImportError:
        return
    for base in nvidia.__path__:
        for sub in os.listdir(base):
            b = os.path.join(base, sub, "bin")
            if os.path.isdir(b):
                try:
                    os.add_dll_directory(b)
                except Exception:
                    pass
                os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")


def load_whisper(status):
    _add_nvidia_dlls()
    from faster_whisper import WhisperModel
    try:
        status(f"Loading whisper {WHISPER_MODEL} on GPU...")
        m = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
        segs, _ = m.transcribe(np.zeros(TARGET_SR, np.float32), language="en", beam_size=1)
        list(segs)  # consume: CUDA/cuDNN errors surface here
        return m, "GPU"
    except Exception as e:
        log("GPU whisper failed:", repr(e))
        status(f"GPU failed ({type(e).__name__}) - using CPU")
        m = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
        return m, "CPU"


# ------------------------------------------------------------------- audio ---
def list_loopback_devices(pa):
    """All WASAPI output devices that can be captured, as (index, display name)."""
    return [(d["index"], d["name"].replace(" [Loopback]", ""))
            for d in pa.get_loopback_device_info_generator()]


def find_loopback(pa, device_index=None):
    """Loopback device by index, or the one matching the Windows default output."""
    import pyaudiowpatch as pyaudio
    if device_index is not None:
        return pa.get_device_info_by_index(device_index)
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    spk = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if spk.get("isLoopbackDevice"):
        return spk
    for lb in pa.get_loopback_device_info_generator():
        if spk["name"] in lb["name"]:
            return lb
    raise RuntimeError("No WASAPI loopback device for default speakers: " + spk["name"])


def pick_device_from_args(devices):
    """--device NAME  -> index of the first loopback device whose name contains NAME."""
    if "--device" in sys.argv:
        want = sys.argv[sys.argv.index("--device") + 1].lower()
        for idx, name in devices:
            if want in name.lower():
                return idx
        log(f"--device '{want}' not found; using default output")
    return None


def resample(x, sr, target=TARGET_SR):
    if sr == target or len(x) == 0:
        return x.astype(np.float32)
    n = int(round(len(x) * target / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


class Segmenter:
    """Energy VAD: groups 30 ms frames into utterances separated by pauses."""

    def __init__(self, emit):
        self.emit = emit
        self.noise = 0.002
        self.speech = False
        self.buf = []
        self.pre = collections.deque(maxlen=max(1, int(PRE_ROLL_SEC / FRAME_SEC)))
        self.silence = 0.0
        self.dur = 0.0

    def feed(self, frame):
        rms = float(np.sqrt(np.mean(frame * frame)) + 1e-9)
        thr = max(ABS_SPEECH_RMS, self.noise * NOISE_MULT)
        voiced = rms > thr
        if not voiced:
            self.noise = max(1e-4, 0.95 * self.noise + 0.05 * rms)

        if self.speech:
            self.buf.append(frame)
            self.dur += FRAME_SEC
            self.silence = 0.0 if voiced else self.silence + FRAME_SEC
            if self.silence >= MIN_SILENCE_SEC:
                self._flush(restart=False)
            elif self.dur >= MAX_UTTERANCE_SEC:
                self._flush(restart=True)
        else:
            self.pre.append(frame)
            if voiced:
                self.speech = True
                self.buf = list(self.pre)
                self.dur = len(self.buf) * FRAME_SEC
                self.silence = 0.0

    def _flush(self, restart):
        audio = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
        voiced_dur = self.dur - self.silence
        self.buf = []
        self.pre.clear()
        self.dur = 0.0
        self.silence = 0.0
        self.speech = restart
        if voiced_dur >= MIN_UTTERANCE_SEC:
            self.emit(audio)


class AudioCapture(threading.Thread):
    def __init__(self, utter_q, status, device_index=None):
        super().__init__(daemon=True)
        self.utter_q = utter_q
        self.status = status
        self.device_index = device_index
        self.stop_flag = threading.Event()
        self.device_name = "?"

    def run(self):
        import pyaudiowpatch as pyaudio
        pa = pyaudio.PyAudio()
        try:
            dev = find_loopback(pa, self.device_index)
            self.device_name = dev["name"]
            sr = int(dev["defaultSampleRate"])
            ch = max(1, int(dev["maxInputChannels"]))
            frames = int(sr * FRAME_SEC)
            seg = Segmenter(self.utter_q.put)
            stream = pa.open(format=pyaudio.paFloat32, channels=ch, rate=sr, input=True,
                             input_device_index=dev["index"], frames_per_buffer=frames)
            self.status(f"Listening: {dev['name']}")
            while not self.stop_flag.is_set():
                data = stream.read(frames, exception_on_overflow=False)
                x = np.frombuffer(data, np.float32)
                if ch > 1:
                    x = x.reshape(-1, ch).mean(axis=1)
                seg.feed(resample(x, sr))
            stream.stop_stream()
            stream.close()
        except Exception as e:
            log("Audio error:", repr(e))
            self.status(f"Audio error: {e}")
        finally:
            pa.terminate()


# ------------------------------------------------------------ transcriber ---
class Transcriber(threading.Thread):
    def __init__(self, utter_q, on_text, status, prompt_hint=""):
        super().__init__(daemon=True)
        self.utter_q = utter_q
        self.on_text = on_text
        self.status = status
        self.prompt_hint = prompt_hint[:600]
        self.backend = "?"
        self.last_latency = 0.0

    def run(self):
        model, self.backend = load_whisper(self.status)
        self.status(f"Whisper ready ({self.backend})")
        while True:
            audio = self.utter_q.get()
            if audio is None:
                break
            t0 = time.time()
            try:
                segs, _ = model.transcribe(
                    audio, language="en", beam_size=1, best_of=1, temperature=0.0,
                    vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
                    condition_on_previous_text=False, without_timestamps=True,
                    initial_prompt=self.prompt_hint or None,
                )
                parts = [s.text.strip() for s in segs
                         if s.no_speech_prob < 0.6 and s.avg_logprob > -1.2]
            except Exception as e:
                log("Transcribe error:", repr(e))
                continue
            self.last_latency = time.time() - t0
            text = " ".join(p for p in parts if p).strip()
            if text and text.lower() not in HALLUCINATIONS:
                self.on_text(text)


# ----------------------------------------------------------------- trigger ---
_qw_re = re.compile(r"^(?:%s)\b" % "|".join(re.escape(w) for w in QUESTION_WORDS))


def is_trigger(text):
    t = text.strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    if _qw_re.match(re.sub(r"^[^a-z]+", "", t)):
        return True
    for n in MY_NAMES:
        if n and re.search(r"\b" + re.escape(n.lower()) + r"\b", t):
            return True
    return False


# ------------------------------------------------------------------ ollama ---
SYSTEM_PROMPT = (
    "You help a university student answer the professor during a live online class. "
    "You get the course notes and the last 60 seconds of what the professor said. "
    "Reply with the answer the student should say out loud: 1 or 2 short sentences, "
    "simple English, no markdown, no bullet points, no preamble, no 'the answer is'. "
    "Prefer facts from the notes. If the transcript is unclear, give your best short guess."
)


def load_notes():
    try:
        with open(NOTES_FILE, encoding="utf-8", errors="ignore") as f:
            notes = f.read().strip()
    except FileNotFoundError:
        notes = ""
    if len(notes) > MAX_NOTES_CHARS:
        notes = notes[:MAX_NOTES_CHARS] + "\n[...notes truncated...]"
    return notes or "(no notes provided)"


def ollama_post(path, body, timeout=90):
    req = urllib.request.Request(OLLAMA_URL + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ollama_warmup():
    try:
        # same num_ctx as ollama_answer, otherwise Ollama reloads the model on the first real question
        ollama_post("/api/generate", {"model": OLLAMA_MODEL, "keep_alive": "30m",
                                      "options": {"num_ctx": 8192}}, timeout=180)
        log("Ollama model loaded:", OLLAMA_MODEL)
    except Exception as e:
        log("Ollama warmup failed:", repr(e))


def ollama_answer(question, context, notes):
    user = (
        f"Course notes:\n{notes}\n\n"
        f"Last 60 seconds of the class (oldest first):\n{context}\n\n"
        f"The professor just said: \"{question}\"\n\n"
        "What should the student answer?"
    )
    data = ollama_post("/api/chat", {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.2, "num_predict": 90, "num_ctx": 8192},
    })
    text = data.get("message", {}).get("content", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"[*_`#>]+", "", text)
    return " ".join(text.split()).strip() or "(empty answer)"


# --------------------------------------------------------------------- GUI ---
class App:
    BG, FG, DIM, ACCENT = "#121417", "#e8e8e8", "#8a8f98", "#4cc2ff"

    def __init__(self):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Lecture Helper")
        self.root.geometry("480x320+40+40")
        self.root.minsize(360, 220)
        self.root.configure(bg=self.BG)
        self.root.attributes("-topmost", True)

        self.ui_q = queue.Queue()
        self.utter_q = queue.Queue()
        self.transcript = collections.deque()      # (timestamp, text)
        self.lock = threading.Lock()
        self.busy = False
        self.last_answer_ts = 0.0
        self.auto = tk.BooleanVar(value=AUTO_ANSWER)

        # ---- device picker: which output device to listen to ----
        import pyaudiowpatch as pyaudio
        from tkinter import ttk
        pa = pyaudio.PyAudio()
        self.devices = list_loopback_devices(pa)              # [(index, name), ...]
        try:
            default_idx = find_loopback(pa)["index"]
        except Exception:
            default_idx = self.devices[0][0] if self.devices else None
        pa.terminate()
        self.device_index = pick_device_from_args(self.devices)
        if self.device_index is None:
            self.device_index = default_idx

        devrow = tk.Frame(self.root, bg=self.BG)
        devrow.pack(fill="x", padx=10, pady=(8, 0))
        tk.Label(devrow, text="Listen to", fg=self.DIM, bg=self.BG, font=("Segoe UI", 9)).pack(side="left")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TCombobox", fieldbackground="#1c2026", background="#2a2e35",
                        foreground=self.FG, arrowcolor=self.FG, selectbackground="#1c2026",
                        selectforeground=self.FG)
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(devrow, textvariable=self.device_var, state="readonly",
                                       style="Dark.TCombobox", font=("Segoe UI", 9),
                                       values=[n for _, n in self.devices])
        for idx, name in self.devices:
            if idx == self.device_index:
                self.device_var.set(name)
        self.device_box.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.device_box.bind("<<ComboboxSelected>>", self._on_device_change)

        top = tk.Frame(self.root, bg=self.BG)
        top.pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(top, text="Heard", fg=self.DIM, bg=self.BG, font=("Segoe UI", 9)).pack(anchor="w")
        self.heard = tk.Label(top, text="...", fg=self.FG, bg=self.BG, font=("Segoe UI", 10),
                              justify="left", anchor="w", wraplength=450)
        self.heard.pack(fill="x")

        mid = tk.Frame(self.root, bg=self.BG)
        mid.pack(fill="both", expand=True, padx=10, pady=(6, 0))
        tk.Label(mid, text="Answer", fg=self.DIM, bg=self.BG, font=("Segoe UI", 9)).pack(anchor="w")
        self.answer = tk.Label(mid, text="Waiting for a question...", fg=self.ACCENT, bg=self.BG,
                               font=("Segoe UI Semibold", 16), justify="left", anchor="nw", wraplength=450)
        self.answer.pack(fill="both", expand=True)

        bot = tk.Frame(self.root, bg=self.BG)
        bot.pack(fill="x", padx=10, pady=8)
        tk.Button(bot, text="Answer now", command=self.answer_now, bg="#1f6feb", fg="white",
                  activebackground="#388bfd", relief="flat", padx=12, pady=4,
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Checkbutton(bot, text="Auto", variable=self.auto, fg=self.FG, bg=self.BG,
                       selectcolor="#2a2e35", activebackground=self.BG,
                       activeforeground=self.FG).pack(side="left", padx=10)
        tk.Button(bot, text="Clear", command=self.clear, bg="#2a2e35", fg=self.FG,
                  relief="flat", padx=8).pack(side="left")
        self.status = tk.Label(bot, text="starting...", fg=self.DIM, bg=self.BG, font=("Segoe UI", 8), anchor="e")
        self.status.pack(side="right", fill="x", expand=True)

        self.root.bind("<Return>", lambda e: self.answer_now())
        self.root.bind("<Configure>", self._on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        if HIDE_FROM_SCREEN_SHARE and "--visible" not in sys.argv:
            self.root.after(300, self._hide_from_capture)

        notes = load_notes()
        self.capture = AudioCapture(self.utter_q, self.set_status, self.device_index)
        self.transcriber = Transcriber(self.utter_q, self.on_text, self.set_status, prompt_hint=notes)
        self.capture.start()
        self.transcriber.start()
        threading.Thread(target=ollama_warmup, daemon=True).start()
        self.root.after(100, self._pump)

    # ---- helpers called from worker threads (thread-safe via queue) ----
    def set_status(self, s):
        log("status:", s)
        self.ui_q.put(("status", s))

    def on_text(self, text):
        now = time.time()
        with self.lock:
            self.transcript.append((now, text))
            while self.transcript and now - self.transcript[0][0] > TRANSCRIPT_WINDOW_SEC:
                self.transcript.popleft()
        log("heard:", text, f"({self.transcriber.last_latency:.2f}s)")
        self.ui_q.put(("heard", text))
        self.ui_q.put(("status", f"{self.transcriber.backend} whisper {self.transcriber.last_latency:.1f}s | listening"))
        if self.auto.get() and is_trigger(text) and now - self.last_answer_ts > ANSWER_COOLDOWN_SEC:
            self.answer_now(trigger=text)

    def answer_now(self, trigger=None):
        if self.busy:
            return
        with self.lock:
            if not self.transcript:
                self.ui_q.put(("answer", "Nothing heard yet."))
                return
            context = "\n".join(t for _, t in self.transcript)
            if trigger is None:
                trigger = self.transcript[-1][1]
        self.busy = True
        self.last_answer_ts = time.time()
        self.ui_q.put(("status", "thinking..."))
        threading.Thread(target=self._answer_worker, args=(trigger, context), daemon=True).start()

    def _answer_worker(self, trigger, context):
        t0 = time.time()
        try:
            ans = ollama_answer(trigger, context, load_notes())
            self.ui_q.put(("answer", ans))
            self.ui_q.put(("status", f"answered in {time.time() - t0:.1f}s | listening"))
            log("answer:", ans)
        except urllib.error.URLError as e:
            self.ui_q.put(("answer", f"Ollama not reachable at {OLLAMA_URL} ({e.reason}). Run: ollama serve"))
        except Exception as e:
            self.ui_q.put(("answer", f"Error: {e}"))
        finally:
            self.busy = False

    # ---- main thread ----
    def _on_device_change(self, _event=None):
        name = self.device_var.get()
        idx = next((i for i, n in self.devices if n == name), None)
        if idx is None or idx == self.device_index:
            return
        self.device_index = idx
        self.capture.stop_flag.set()
        self.capture.join(timeout=2)
        self.capture = AudioCapture(self.utter_q, self.set_status, idx)
        self.capture.start()
        self.root.focus()  # drop focus from the combobox so Enter still means "Answer now"

    def _pump(self):
        try:
            while True:
                kind, val = self.ui_q.get_nowait()
                if kind == "status":
                    self.status.config(text=val)
                elif kind == "heard":
                    self.heard.config(text=val)
                elif kind == "answer":
                    self.answer.config(text=val)
        except queue.Empty:
            pass
        self.root.after(100, self._pump)

    def _on_resize(self, e):
        if e.widget is self.root:
            w = max(200, e.width - 30)
            self.heard.config(wraplength=w)
            self.answer.config(wraplength=w)

    def _hide_from_capture(self):
        try:
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            WDA_EXCLUDEFROMCAPTURE = 0x11
            if not ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
                log("SetWindowDisplayAffinity failed (needs Windows 10 2004+)")
        except Exception as e:
            log("hide from capture failed:", repr(e))

    def clear(self):
        with self.lock:
            self.transcript.clear()
        self.heard.config(text="...")
        self.answer.config(text="Waiting for a question...")

    def quit(self):
        self.capture.stop_flag.set()
        self.utter_q.put(None)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ------------------------------------------------------------------- check ---
def self_check():
    ok = True
    print("== audio ==")
    try:
        import pyaudiowpatch as pyaudio
        pa = pyaudio.PyAudio()
        devices = list_loopback_devices(pa)
        for idx, name in devices:
            print(f"  [{idx}] {name}")
        dev = find_loopback(pa, pick_device_from_args(devices))
        print("selected:", dev["name"], "| rate", int(dev["defaultSampleRate"]), "| ch", dev["maxInputChannels"])
        pa.terminate()
    except Exception as e:
        ok = False
        print("FAIL:", repr(e))

    print("== whisper ==")
    try:
        t0 = time.time()
        model, backend = load_whisper(print)
        print("backend:", backend, f"| load {time.time() - t0:.1f}s")
        t0 = time.time()
        segs, _ = model.transcribe(np.random.randn(TARGET_SR * 3).astype(np.float32) * 0.01, language="en", beam_size=1)
        list(segs)
        print(f"3s transcribe: {time.time() - t0:.2f}s")
    except Exception as e:
        ok = False
        print("FAIL:", repr(e))

    print("== ollama ==")
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
        print("models:", names)
        if OLLAMA_MODEL not in names:
            ok = False
            print(f"FAIL: {OLLAMA_MODEL} missing. Run: ollama pull {OLLAMA_MODEL}")
        else:
            t0 = time.time()
            ans = ollama_answer("What is the capital of France?", "Today we talk about Europe.", "France: capital Paris.")
            print(f"answer ({time.time() - t0:.1f}s):", ans)
    except Exception as e:
        ok = False
        print("FAIL:", repr(e), "| is Ollama running? (ollama serve)")

    print("== notes ==")
    print(NOTES_FILE, "|", "found" if os.path.exists(NOTES_FILE) else "MISSING (create it)")
    print("== triggers ==")
    for s in ["What is a pointer?", "Explain the difference.", "so we move on to chapter two", "Ahmed, what do you think"]:
        print(f"  {is_trigger(s)!s:5} <- {s}")
    print("\nRESULT:", "OK" if ok else "PROBLEMS ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(self_check())
    App().run()
