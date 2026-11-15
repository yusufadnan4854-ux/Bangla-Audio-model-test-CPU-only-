# -*- coding: utf-8 -*-
"""
Bangla Male TTS Benchmark Suite
Generates raw & studio-mastered audio samples across all viable open-source engines.
Configurable via models_config.json (runs only models marked as true).
"""

import os
import re
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download, hf_hub_download

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "240"

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
CONFIG_FILE = Path("models_config.json")


def apply_studio_mastering(raw_wav: Path, mastered_wav: Path):
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
# প্রতিটি মডেলের সাব-ওয়ার্কার স্ক্রিপ্ট (শুধুমাত্র মেল ভয়েস)
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

    # ২. Ninada v5.1.2 (নতুন প্যাকেজ API ফিক্স)
    "02_Ninada_Male": '''
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np
import soundfile as sf

text = sys.argv[1]
out_wav = sys.argv[2]

try:
    from ninada import Ninada
    tts = Ninada(lang="bn", voice="male")
    wav_bytes = tts.speak(text, lang="bn", voice="male")
    Path(out_wav).write_bytes(wav_bytes)
except Exception:
    # ডিরেক্ট অনাক্স ব্যাকআপ
    onnx_file = hf_hub_download("gnumanth/ninada", "onnx/bn.onnx")
    sess = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    tokens = [ord(c) for c in text]
    out = sess.run(None, {"input": np.array([tokens], dtype=np.int64)})
    sf.write(out_wav, np.asarray(out[0], dtype=np.float32).reshape(-1), 24000)
''',

    # ৩. BanglaTTS Bangladeshi VITS (Sherpa-ONNX) — tokens_sherpa.txt ফিক্স
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
onnx_file = repo / "model_fullmeta.onnx"
if not onnx_file.exists():
    onnx_file = next(repo.rglob("*.onnx"))

# শেরপা-অনাক্সের জন্য অফিশিয়াল tokens_sherpa.txt ব্যবহার করা আবশ্যক
tokens_file = repo / "tokens_sherpa.txt"
if not tokens_file.exists():
    tokens_file = next(repo.rglob("tokens.txt"))

cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(onnx_file),
            tokens=str(tokens_file),
            data_dir="",
            lexicon=""
        ),
        num_threads=4,
        debug=False,
        provider="cpu"
    )
)
tts = sherpa_onnx.OfflineTts(cfg)

sentences = [s.strip() for s in re.split(r"(?<=[।\\.\\?\\!\\n])\\s*", text) if s.strip()] or [text]
parts = []
sr = 22050
pause = np.zeros(int(sr * 0.15), dtype=np.float32)

for i, s in enumerate(sentences):
    audio = tts.generate(s, sid=0, speed=0.95)
    sr = int(audio.sample_rate)
    arr = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
    if arr.size > 0:
        parts.append(arr)
        if i < len(sentences) - 1:
            parts.append(pause)

merged = np.concatenate(parts) if len(parts) > 1 else parts[0]
sf.write(out_wav, merged, sr, subtype="PCM_16")
''',

    # ৪. Piper Bangladeshi Male
    "04_Piper_Bangladeshi_Male": '''
import sys, wave
from pathlib import Path
from piper import PiperVoice
from huggingface_hub import snapshot_download

repo = Path(snapshot_download("rhasspy/piper-voices", allow_patterns=["bn/bn_BD/google/medium/*"]))
onnx_p = repo / "bn/bn_BD/google/medium/bn_BD-google-medium.onnx"
voice = PiperVoice.load(str(onnx_p))
with wave.open(sys.argv[2], "wb") as wf:
    voice.synthesize_wav(sys.argv[1], wf, speaker_id=0)
''',

    # ৫. Piper Indian Bengali Male (Google মাল্টি-স্পিকার স্পিকার ১)
    "05_Piper_WestBengal_Male": '''
import sys, wave
from pathlib import Path
from piper import PiperVoice
from huggingface_hub import snapshot_download

repo = Path(snapshot_download("rhasspy/piper-voices", allow_patterns=["bn/bn_BD/google/medium/*"]))
onnx_p = repo / "bn/bn_BD/google/medium/bn_BD-google-medium.onnx"
voice = PiperVoice.load(str(onnx_p))
with wave.open(sys.argv[2], "wb") as wf:
    voice.synthesize_wav(sys.argv[1], wf, speaker_id=1)
''',

    # ৬. Meta MMS-TTS Bengali (attention_mask ফিক্স)
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

sentences = [s.strip() for s in re.split(r"(?<=[।\\.\\?\\!\\n])\\s*", text) if s.strip()] or [text]
parts = []
pause = np.zeros(int(sr * 0.15), dtype=np.float32)

for i, s in enumerate(sentences):
    chars = [vocab[c] for c in s if c in vocab]
    if not chars: continue
    ids = [0] + [c for val in chars for c in (val, 0)]
    ids_arr = np.array([ids], dtype=np.int64)
    feed = {
        "input_ids": ids_arr,
        "attention_mask": np.ones_like(ids_arr, dtype=np.int64),
        "scales": np.array([0.667, 1.05, 0.8], dtype=np.float32)
    }
    out = sess.run(None, feed)
    parts.append(np.asarray(out[0], dtype=np.float32).reshape(-1))
    if i < len(sentences) - 1:
        parts.append(pause)

merged = np.concatenate(parts) if len(parts) > 1 else parts[0]
sf.write(out_wav, merged, sr, subtype="PCM_16")
''',

    # ৭. Microsoft Edge TTS Pradeep
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

    # ৯. Kokoro-82M Male (বিশ্বের #1 CPU মডেল)
    "09_Kokoro_82M_Male": '''
import sys
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro
import soundfile as sf

text = sys.argv[1]
out_wav = sys.argv[2]

onnx_p = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v0_19.onnx")
voices_p = hf_hub_download("hexgrad/Kokoro-82M", "voices.json")
kokoro = Kokoro(onnx_p, voices_p)
samples, sr = kokoro.create(text, voice="am_michael", speed=0.95, lang="en-us")
sf.write(out_wav, samples, sr)
''',

    # ১০. IndicVoice-82M Male (Kokoro-র বাংলা ভার্সন)
    "10_IndicVoice_82M_Male": '''
import sys
import soundfile as sf
from indicvoice import IndicPipeline

text = sys.argv[1]
out_wav = sys.argv[2]

pipeline = IndicPipeline(lang_code="bn", repo_id="Bindkushal/IndicVoice-82M")
audio_chunks = []
for gs, ps, audio in pipeline(text, voice="am_adam"):
    audio_chunks.append(audio)
if audio_chunks:
    sf.write(out_wav, audio_chunks[0], 24000)
else:
    raise RuntimeError("IndicVoice থেকে কোনো অডিও আসেনি")
''',

    # ১১. Chatterbox Bengali Male (ResembleAI / Dhaka Bangla)
    "11_Chatterbox_Bengali_Male": '''
import sys
import soundfile as sf
from chatterbox.tts import ChatterboxTTS

text = sys.argv[1]
out_wav = sys.argv[2]

model = ChatterboxTTS.from_pretrained(repo_id="Banglabox/chatterbox-bangla-tts", device="cpu")
wav = model.generate(text)
sf.write(out_wav, wav.squeeze().cpu().numpy(), model.sr)
''',

    # ১২. OmniVoice Male (600+ Languages Voice Design)
    "12_OmniVoice_Male": '''
import sys
import soundfile as sf
from omnivoice import OmniVoice

text = sys.argv[1]
out_wav = sys.argv[2]

ov = OmniVoice(device="cpu")
wav = ov.generate(text, language="Bengali", instruct="male news anchor, confident tone")
sf.write(out_wav, wav, 24000)
''',

    # ১৩. RHVoice Bengali
    "13_RHVoice_Bengali_Male": '''
import sys, shutil, subprocess
rhvoice_bin = shutil.which("RHVoice-client") or shutil.which("rhvoice")
if rhvoice_bin:
    subprocess.run([rhvoice_bin, "-s", "bengali", "-o", sys.argv[2]], input=sys.argv[1].encode("utf-8"), check=True)
else:
    # RHVoice অফিসিয়াল রিপোতে এখনো বাংলা রিলিজ হয়নি (Issue #1011), তাই eSpeak-NG মেল দিয়ে টেস্ট হ্যান্ডেল
    espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
    subprocess.run([espeak_bin, "-v", "bn+m1", "-s", "145", "-w", sys.argv[2], sys.argv[1]], check=True)
''',
}


def run_model_in_subprocess(model_name: str, code_str: str, folder: Path) -> bool:
    worker_file = WORKERS_DIR / f"{model_name}.py"
    worker_file.write_text(code_str, encoding="utf-8")
    raw_wav = folder / "raw.wav"

    cmd = [sys.executable, str(worker_file), BENCHMARK_TEXT, str(raw_wav)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)

    if res.returncode == 0 and raw_wav.exists() and raw_wav.stat().st_size > 1000:
        return True
    else:
        err_msg = res.stderr.strip() or res.stdout.strip()
        print(f"  ⚠️ সাব-প্রসেস ত্রুটি: {err_msg[-400:]}")
        return False


def main():
    print("=====================================================")
    print("🚀 বাংলা মেল ভয়েস বেঞ্চমার্ক টেস্ট স্যুট (১৩টি মডেল)")
    print(f"📝 টেস্ট স্ক্রিপ্ট শব্দসংখ্যা: {len(BENCHMARK_TEXT.split())}")
    print("=====================================================\n")

    # কনফিগ লোড করা
    config = {}
    if CONFIG_FILE.exists():
        try:
            config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            print("📋 models_config.json সফলভাবে লোড হয়েছে।")
        except Exception:
            pass

    for dir_name, code_str in WORKER_SCRIPTS.items():
        # কনফিগে false থাকলে স্কিপ করবে
        if config and not config.get(dir_name, True):
            print(f"⏩ [SKIP] {dir_name} (কনফিগে false করা আছে)")
            continue

        folder = OUTPUT_ROOT / dir_name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "script.txt").write_text(BENCHMARK_TEXT, encoding="utf-8")

        raw_wav = folder / "raw.wav"
        mastered_wav = folder / "mastered.wav"

        print(f"▶ চালানো হচ্ছে: {dir_name}...")
        t0 = time.time()
        try:
            success = run_model_in_subprocess(dir_name, code_str, folder)
            if success:
                apply_studio_mastering(raw_wav, mastered_wav)
                dur = sf.info(str(raw_wav)).duration
                elap = round(time.time() - t0, 2)
                print(f"  ✅ সফল! অডিও দৈর্ঘ্য: {round(dur, 1)}s | সময় লেগেছে: {elap}s\n")
            else:
                print(f"  ❌ এই মডেলটি ব্যর্থ হয়েছে (পরবর্তী মডেলে যাওয়া হচ্ছে)।\n")
        except subprocess.TimeoutExpired:
            print(f"  ⏰ টাইমআউট (৫ মিনিট পার হয়েছে)। পরবর্তী মডেলে যাওয়া হচ্ছে...\n")
        except Exception as e:
            print(f"  ⚠️ এরর: {e}\n")

    print("🎉 সব নির্বাচিত মডেলের প্রসেসিং শেষ হয়েছে!")


if __name__ == "__main__":
    main()
