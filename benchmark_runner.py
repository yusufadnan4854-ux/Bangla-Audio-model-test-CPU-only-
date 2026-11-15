# -*- coding: utf-8 -*-
"""
Bangla Male TTS Benchmark Suite
Generates raw & studio-mastered audio samples across all viable open-source engines.
Runs each model in isolated sub-processes so one model crash never stops the pipeline.
"""

import os
import re
import sys
import json
import time
import shutil
import inspect
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"

# টেস্ট করার জন্য প্রমিত বাংলা স্ক্রিপ্ট (ইউটিউব ও সার্কুলার স্টাইল)
BENCHMARK_TEXT = (
    "আসসালামু আলাইকুম প্রিয় চাকরিপ্রার্থী ভাই ও বোনেরা। আজকের ভিডিওতে আপনাদের সবাইকে আন্তরিক স্বাগতম। "
    "আজকে আমরা আলোচনা করব নতুন একটি নিয়োগ বিজ্ঞপ্তি নিয়ে, যেখানে বিভিন্ন গুরুত্বপূর্ণ পদে যোগ্য ও দক্ষ জনবল নিয়োগ দেওয়া হবে। "
    "শিক্ষাগত যোগ্যতা, অভিজ্ঞতা এবং সরকারি বিধি মোতাবেক বেতন স্কেল ও সুযোগ-সুবিধা প্রযোজ্য হবে। "
    "সম্পূর্ণ তথ্য সতর্কতার সাথে যাচাই করে নির্ধারিত সময়ের মধ্যে আবেদন সম্পন্ন করার জন্য পরামর্শ দেওয়া হলো। ধন্যবাদ।"
)

OUTPUT_ROOT = Path("tts_results")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
WORKERS_DIR = Path("/tmp/tts_subworkers")
WORKERS_DIR.mkdir(parents=True, exist_ok=True)


def apply_studio_mastering(raw_wav: Path, mastered_wav: Path):
    """FFmpeg প্রো-অডিও ফিল্টার: ডিপ বেইস, ডিজিটাল হার্শনেস কাট, ডায়নামিক কম্প্রেশন ও লাউডনেস নরমালাইজেশন"""
    if not shutil.which("ffmpeg"):
        shutil.copy2(raw_wav, mastered_wav)
        return

    cmd = [
        "ffmpeg", "-y", "-i", str(raw_wav),
        "-af",
        "highpass=f=80,"
        "equalizer=f=250:t=q:w=1.2:g=2.5,"
        "equalizer=f=4500:t=q:w=1.5:g=-3.5,"
        "acompressor=threshold=-18dB:ratio=2.5:attack=10:release=100,"
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(mastered_wav)
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode != 0 or not mastered_wav.exists() or mastered_wav.stat().st_size < 1000:
        shutil.copy2(raw_wav, mastered_wav)


# =========================================================================
# সাব-প্রসেস ওয়ার্কার স্ক্রিপ্টসমূহ
# =========================================================================

WORKER_SCRIPTS = {
    # ১. Indic-Tiny-TTS
    "01_Indic_Tiny_TTS_Male": '''
import sys
from huggingface_hub import snapshot_download
repo = snapshot_download("sulabhkatiyar/indic-tiny-tts-bengali", ignore_patterns=["*.safetensors", "samples/*", "samples/**"])
if repo not in sys.path:
    sys.path.insert(0, repo)
from indic_tiny_tts import TinyTTSOnnx
tts = TinyTTSOnnx.from_pretrained(repo)
wav = tts.synthesize(sys.argv[1], voice="male", speed=0.95)
tts.save_wav(wav, sys.argv[2])
''',

    # ২. Ninada v5.1.2
    "02_Ninada_Male": '''
import sys
from ninada import synthesize
synthesize(text=sys.argv[1], output_file=sys.argv[2], language="bn", speaker=0)
''',

    # ৩. BanglaTTS Bangladeshi VITS (Sherpa-ONNX) — ডুপ্লিকেট টোকেন ক্লিন সহ
    "03_BanglaTTS_Sherpa_BD_Male": '''
import sys, re
from pathlib import Path
import numpy as np
import soundfile as sf
import sherpa_onnx
from huggingface_hub import snapshot_download

text = sys.argv[1]
out_wav = sys.argv[2]

repo = Path(snapshot_download("demoqqq/banglatts-kiki-hasan-big"))
onnx_file = next(repo.rglob("*.onnx"))
raw_tokens = next(repo.rglob("tokens.txt"))

# Sherpa-ONNX ডুপ্লিকেট টোকেন C++ ক্র্যাশ ফিক্স
clean_tokens = Path("/tmp/cleaned_tokens.txt")
seen_ids = set()
clean_lines = []
for line in raw_tokens.read_text(encoding="utf-8").splitlines():
    parts = line.strip().split()
    if len(parts) >= 2:
        tid = parts[-1]
        if tid not in seen_ids:
            seen_ids.add(tid)
            clean_lines.append(line)
clean_tokens.write_text("\\n".join(clean_lines), encoding="utf-8")

cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(onnx_file),
            tokens=str(clean_tokens),
            data_dir="",
            lexicon=""
        ),
        num_threads=4,
        debug=False,
        provider="cpu"
    )
)
tts = sherpa_onnx.OfflineTts(cfg)

parts_text = [s.strip() for s in re.split(r"(?<=[।\\.\\?\\!\\n])\\s*", text) if s.strip()] or [text]
parts = []
sr = 22050
pause = np.zeros(int(sr * 0.15), dtype=np.float32)

for i, s in enumerate(parts_text):
    audio = tts.generate(s, sid=0, speed=0.95)
    sr = int(audio.sample_rate)
    arr = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
    if arr.size > 0:
        parts.append(arr)
        if i < len(parts_text) - 1:
            parts.append(pause)

merged = np.concatenate(parts) if len(parts) > 1 else parts[0]
sf.write(out_wav, merged, sr, subtype="PCM_16")
''',

    # ৪. Piper Bangladeshi
    "04_Piper_Bangladeshi_Male": '''
import sys, wave
from pathlib import Path
from piper import PiperVoice
from huggingface_hub import snapshot_download

repo = Path(snapshot_download("rhasspy/piper-voices", allow_patterns=["bn/bn_BD/google/medium/bn_BD-google-medium.onnx*"]))
onnx_p = repo / "bn/bn_BD/google/medium/bn_BD-google-medium.onnx"
voice = PiperVoice.load(str(onnx_p))
with wave.open(sys.argv[2], "wb") as wf:
    voice.synthesize_wav(sys.argv[1], wf)
''',

    # ৫. Piper Indian Bengali
    "05_Piper_WestBengal_Male": '''
import sys, wave
from pathlib import Path
from piper import PiperVoice
from huggingface_hub import snapshot_download

repo = Path(snapshot_download("rhasspy/piper-voices", allow_patterns=["bn/bn_IN/biswas/medium/bn_IN-biswas-medium.onnx*"]))
onnx_p = repo / "bn/bn_IN/biswas/medium/bn_IN-biswas-medium.onnx"
voice = PiperVoice.load(str(onnx_p))
with wave.open(sys.argv[2], "wb") as wf:
    voice.synthesize_wav(sys.argv[1], wf)
''',

    # ৬. Meta MMS-TTS Bengali
    "06_Meta_MMS_Bengali_Male": '''
import sys, json, re
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort
from huggingface_hub import snapshot_download

text = sys.argv[1]
out_wav = sys.argv[2]

repo = Path(snapshot_download("naklitechie/mms-tts-bn-ONNX"))
vocab = json.loads((repo / "vocab.json").read_text(encoding="utf-8"))
cfg = json.loads((repo / "config.json").read_text(encoding="utf-8"))
sess = ort.InferenceSession(str(repo / "model.onnx"), providers=["CPUExecutionProvider"])
sr = int(cfg.get("sampling_rate", 16000))

parts_text = [s.strip() for s in re.split(r"(?<=[।\\.\\?\\!\\n])\\s*", text) if s.strip()] or [text]
parts = []
pause = np.zeros(int(sr * 0.15), dtype=np.float32)

for i, s in enumerate(parts_text):
    chars = [vocab[c] for c in s if c in vocab]
    if not chars: continue
    ids = [0] + [c for val in chars for c in (val, 0)]
    feed = {
        "input_ids": np.array([ids], dtype=np.int64),
        "scales": np.array([0.667, 1.05, 0.8], dtype=np.float32)
    }
    out = sess.run(None, feed)
    parts.append(np.asarray(out[0], dtype=np.float32).reshape(-1))
    if i < len(parts_text) - 1:
        parts.append(pause)

merged = np.concatenate(parts) if len(parts) > 1 else parts[0]
sf.write(out_wav, merged, sr, subtype="PCM_16")
''',

    # ৭. Microsoft Edge TTS
    "07_Edge_TTS_PradeepNeural_Male": '''
import sys, asyncio, subprocess
from pathlib import Path
import edge_tts

text = sys.argv[1]
out_wav = sys.argv[2]
raw_mp3 = Path(out_wav).with_suffix(".mp3")

async def _make():
    comm = edge_tts.Communicate(text, "bn-BD-PradeepNeural", rate="-6%")
    await comm.save(str(raw_mp3))

asyncio.run(_make())
subprocess.run(["ffmpeg", "-y", "-i", str(raw_mp3), "-ar", "24000", "-ac", "1", out_wav], check=True)
if raw_mp3.exists(): raw_mp3.unlink()
''',

    # ৮. eSpeak-NG Bengali
    "08_eSpeak_NG_Bengali_Male": '''
import sys, shutil, subprocess
espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
if espeak_bin:
    subprocess.run([espeak_bin, "-v", "bn+m3", "-s", "145", "-w", sys.argv[2], sys.argv[1]], check=True)
else:
    raise RuntimeError("espeak binary not found")
''',
}


def run_model_in_subprocess(model_name: str, code_str: str, folder: Path) -> bool:
    worker_file = WORKERS_DIR / f"{model_name}.py"
    worker_file.write_text(code_str, encoding="utf-8")
    raw_wav = folder / "raw.wav"

    cmd = [sys.executable, str(worker_file), BENCHMARK_TEXT, str(raw_wav)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if res.returncode == 0 and raw_wav.exists() and raw_wav.stat().st_size > 1000:
        return True
    else:
        err_msg = res.stderr.strip() or res.stdout.strip()
        print(f"  ⚠️ সাব-প্রসেস ত্রুটি: {err_msg[-400:]}")
        return False


def main():
    print("=====================================================")
    print("🚀 বাংলা মেল ভয়েস বেঞ্চমার্ক টেস্ট শুরু হচ্ছে...")
    print(f"📝 টেস্ট স্ক্রিপ্ট শব্দসংখ্যা: {len(BENCHMARK_TEXT.split())}")
    print("=====================================================\n")

    for dir_name, code_str in WORKER_SCRIPTS.items():
        folder = OUTPUT_ROOT / dir_name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "script.txt").write_text(BENCHMARK_TEXT, encoding="utf-8")

        raw_wav = folder / "raw.wav"
        mastered_wav = folder / "mastered.wav"

        print(f"▶ চালানো হচ্ছে: {dir_name}...")
        t0 = time.time()
        success = run_model_in_subprocess(dir_name, code_str, folder)

        if success:
            apply_studio_mastering(raw_wav, mastered_wav)
            dur = sf.info(str(raw_wav)).duration
            elap = round(time.time() - t0, 2)
            print(f"  ✅ সফল! অডিও দৈর্ঘ্য: {round(dur, 1)}s | সময় লেগেছে: {elap}s\n")
        else:
            print(f"  ❌ এই মডেলটি ব্যর্থ হয়েছে (পরবর্তী মডেলে যাওয়া হচ্ছে)।\n")

    print("🎉 সব মডেলের টেস্টিং ও প্রসেসিং শেষ হয়েছে!")


if __name__ == "__main__":
    main()
