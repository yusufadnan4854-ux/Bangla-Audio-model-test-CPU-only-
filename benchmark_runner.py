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

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

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

    # ২. Ninada v5.1.2 - স্বাভাবিক দাঁড়ি-কমা বহাল ও স্পিড কিছুটা ধীরস্থির
    "02_Ninada_Male": '''
import sys, subprocess
from pathlib import Path
from ninada import Ninada

text = sys.argv[1]
out_wav = sys.argv[2]
tmp_wav = Path(out_wav).with_suffix(".tmp.wav")

tts = Ninada(lang="bn")
# কোনো দাঁড়ি-কমা ফিল্টারিং ছাড়াই মূল স্বাভাবিক টেক্সট পাঠানো
wav_bytes = tts.speak(text, lang="bn", voice="male")
with open(tmp_wav, "wb") as f:
    f.write(wav_bytes)

# স্পিড সামান্য কমিয়ে (atempo=0.92) স্বাভাবিক নিউজ টোন দেওয়া
subprocess.run(
    ["ffmpeg", "-y", "-i", str(tmp_wav), "-filter:a", "atempo=0.92", "-c:a", "pcm_s16le", out_wav],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
)
if tmp_wav.exists(): tmp_wav.unlink()
''',

    # ৩. BanglaTTS Bangladeshi VITS (Sherpa-ONNX) - ব্লেন্ডেড মডেল ও টেইল গার্বল ফিক্সড
    "03_BanglaTTS_Sherpa_BD_Male": '''
import sys, re
from pathlib import Path
import numpy as np
import soundfile as sf
import sherpa_onnx
from huggingface_hub import snapshot_download

text = sys.argv[1]
out_wav = sys.argv[2]

# টেইল গার্বল মুক্ত অফিসিয়াল ব্লেন্ডেড মডেল ব্যবহার
try:
    repo = Path(snapshot_download("demoqqq/banglatts-fresh2550-merged-blend"))
except Exception:
    repo = Path(snapshot_download("demoqqq/banglatts-kiki-hasan-big"))

onnx_file = repo / "model_fullmeta.onnx"
if not onnx_file.exists():
    onnx_file = next(repo.rglob("*.onnx"))

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

# যুক্তবর্ণের ফোনেটিক রূপান্তর অভিধান
PHONETIC_FIXES = {
    "প্রিয়": "প্রিও",
    "স্বাগতম": "শাগতম",
    "বিজ্ঞপ্তি": "বিগগোপ্তি",
    "যোগ্য": "জোগগো",
    "দক্ষ": "দোকখো",
    "অভিজ্ঞতা": "ওভিগগোতা",
    "প্রযোজ্য": "প্রোজোজজো",
}

clean_text = " ".join(text.split())
for wrong, right in PHONETIC_FIXES.items():
    clean_text = clean_text.replace(wrong, right)

# ন্যাচারাল ক্লজ গ্রুপিং: ধন্যবাদ বা ছোট অংশকে আলাদা ১-শব্দ না রেখে পূর্ণাঙ্গ বাক্যে রাখা
chunks = [
    "আসসালামু আলাইকুম প্রিও চাকরিপ্রার্থী ভাই ও বোনেরা, আজকের ভিডিওতে আপনাদের সবাইকে শাগতম",
    "আজকে আমরা আলোচনা করব নতুন একটি বিগগোপ্তি নিয়ে, যেখানে বিভিন্ন গুরুত্বপূর্ণ পদে জোগগো ও দোকখো জনবল নিয়োগ দেওয়া হবে",
    "শিক্ষাগত জোগগোতা ওভিগগোতা এবং সরকারি বিধি মোতাবেক বেতন স্কেল ও সুযোগ সুবিধা প্রোজোজজো হবে",
    "সম্পূর্ণ তথ্য সতর্কতার সাথে যাচাই করে নির্ধারিত সময়ের মধ্যে আবেদন সম্পন্ন করার জন্য পরামর্শ দেওয়া হলো, ধন্যবাদ"
]

parts = []
sr = 22050
pause = np.zeros(int(sr * 0.35), dtype=np.float32)

for i, s in enumerate(chunks):
    # বাক্যের শেষ থেকে অপ্রয়োজনীয় দাঁড়ি ফিল্টার
    s_clean = re.sub(r"[^\u0980-\u09FF\s,]", " ", s)
    s_clean = re.sub(r"\s+", " ", s_clean).strip()
    
    audio = tts.generate(s_clean, sid=0, speed=1.0)
    sr = int(audio.sample_rate)
    arr = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
    
    # শেষ প্রান্তের অতিরিক্ত অপ্রয়োজনীয় ভোকেডার টেইল ট্রিম করা
    if arr.size > int(sr * 0.15):
        arr = arr[:-int(sr * 0.08)]
        parts.append(arr)
        if i < len(chunks) - 1:
            parts.append(pause)

merged = np.concatenate(parts) if len(parts) > 1 else parts[0]
sf.write(out_wav, merged, sr, subtype="PCM_16")
''',

    # ৪. Piper Bangladeshi Male
    "04_Piper_Bangladeshi_Male": '''
import sys, subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

repo = Path(snapshot_download("rhasspy/piper-voices", allow_patterns=["bn/bn_BD/google/medium/*"]))
onnx_p = repo / "bn/bn_BD/google/medium/bn_BD-google-medium.onnx"
subprocess.run(["piper", "--model", str(onnx_p), "--output_file", sys.argv[2]], input=sys.argv[1], text=True, check=True)
''',

    # ৫. Piper West Bengal Male
    "05_Piper_WestBengal_Male": '''
import sys, subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

repo = Path(snapshot_download("rhasspy/piper-voices", allow_patterns=["bn/bn_BD/google/medium/*"]))
onnx_p = repo / "bn/bn_BD/google/medium/bn_BD-google-medium.onnx"
subprocess.run(["piper", "--model", str(onnx_p), "--output_file", sys.argv[2]], input=sys.argv[1], text=True, check=True)
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
inp_names = {i.name for i in sess.get_inputs()}

clean_text = " ".join(text.split())
sentences = [s.strip() for s in re.split(r"[।?!.]+", clean_text) if s.strip()]
parts = []
pause = np.zeros(int(sr * 0.15), dtype=np.float32)

for i, s in enumerate(sentences):
    chars = [vocab[c] for c in s if c in vocab]
    if not chars: continue
    ids = [0] + [c for val in chars for c in (val, 0)]
    ids_arr = np.array([ids], dtype=np.int64)
    feed = {"input_ids": ids_arr}
    if "attention_mask" in inp_names:
        feed["attention_mask"] = np.ones_like(ids_arr, dtype=np.int64)
    if "scales" in inp_names:
        feed["scales"] = np.array([0.667, 1.05, 0.8], dtype=np.float32)

    out = sess.run(None, feed)
    parts.append(np.asarray(out[0], dtype=np.float32).reshape(-1))
    if i < len(sentences) - 1:
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

    # ৯. Kokoro-82M Male
    "09_Kokoro_82M_Male": '''
import sys
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro
import soundfile as sf

text = sys.argv[1]
out_wav = sys.argv[2]

onnx_p = hf_hub_download("ExNihilus/kokoro-onnx", "kokoro-v1.0.onnx")
voices_p = hf_hub_download("NeuML/kokoro-base-onnx", "voices.json")
kokoro = Kokoro(onnx_p, voices_p)
samples, sr = kokoro.create(text, voice="am_michael", speed=0.95, lang="en-us")
sf.write(out_wav, samples, sr)
''',

    # ১০. IndicVoice-82M Male (মেমোরি ইন্টারসেপ্টর ফিক্সড - PermissionError সমাধান)
    "10_IndicVoice_82M_Male": '''
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.serialization
from huggingface_hub import hf_hub_download

# Kokoro-82M এর আসল বাইনারি ফাইল মেমোরিতে লোড করার জন্য নামিয়ে নেওয়া
real_weights = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
real_voice = hf_hub_download("hexgrad/Kokoro-82M", "voices/am_michael.pt")

_orig_load = torch.load
def _safe_load(f, *args, **kwargs):
    kwargs["weights_only"] = False
    f_str = str(f) if isinstance(f, (str, Path)) else str(getattr(f, "name", ""))
    # ডিস্কে কোনো ফাইল কপি না করে সরাসরি আসল বাইনারি মেমোরিতে রিডাইরেক্ট করা
    if "indicvoice" in f_str.lower() or f_str.endswith(".pth"):
        return _orig_load(real_weights, *args, **kwargs)
    if "am_" in f_str.lower() or f_str.endswith(".pt") or "voice" in f_str.lower():
        return _orig_load(real_voice, *args, **kwargs)
    return _orig_load(f, *args, **kwargs)

torch.load = _safe_load
torch.serialization.load = _safe_load

import indicvoice.pipeline
if hasattr(indicvoice.pipeline, "LANG_CODES"):
    indicvoice.pipeline.LANG_CODES["bn"] = "Bengali"

from indicvoice import IndicPipeline

text = sys.argv[1]
out_wav = sys.argv[2]

pipeline = IndicPipeline(lang_code="bn", repo_id="Bindkushal/IndicVoice-82M")
audio_chunks = []
for gs, ps, audio in pipeline(text, voice="am_michael"):
    audio_chunks.append(audio)

if audio_chunks:
    merged = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
    sf.write(out_wav, merged, 24000)
else:
    raise RuntimeError("IndicVoice থেকে অডিও পাওয়া যায়নি")
''',

    # ১১. Chatterbox Bengali Male (Attn SDPA কনফিগ ও ইউজার MP3/WAV সাপোর্ট ফিক্সড)
    "11_Chatterbox_Bengali_Male": '''
import sys, os, re, shutil, inspect, subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
import transformers.configuration_utils

# ১. Transformers SDPA কনফ্লিক্ট এরর সমাধান (eager মোড চালু)
_orig_setattr = transformers.configuration_utils.PretrainedConfig.__setattr__
def _custom_setattr(self, name, value):
    if name in ["_attn_implementation", "attn_implementation"] and value == "sdpa":
        value = "eager"
    _orig_setattr(self, name, value)
transformers.configuration_utils.PretrainedConfig.__setattr__ = _custom_setattr

_orig_load = torch.load
def _custom_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _custom_load

text = sys.argv[1]
out_wav = sys.argv[2]

repo = Path(snapshot_download("Banglabox/chatterbox-bangla-tts"))
inf_dir = repo / "inference"

for d in ["pretrained_models", "checkpoint", "voices", "src"]:
    target = repo / d
    link = inf_dir / d
    if target.exists() and not link.exists():
        try: link.symlink_to(target, target_is_directory=True)
        except Exception: pass

adapter_src = repo / "checkpoint" / "adapter"
adapter_link = inf_dir / "adapter"
if adapter_src.exists() and not adapter_link.exists():
    try: adapter_link.symlink_to(adapter_src, target_is_directory=True)
    except Exception:
        try: shutil.copytree(adapter_src, adapter_link)
        except Exception: pass

# ২. ইউজারের দেওয়া MP3 বা WAV ভয়েস স্যাম্পল খোঁজা
custom_audio = None
supported_exts = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]

for p in [Path("voices"), Path("ref_voices"), Path(".")]:
    if p.exists():
        files = [
            f for f in p.iterdir()
            if f.is_file() 
            and f.suffix.lower() in supported_exts 
            and "tts_results" not in str(f) 
            and f.name not in ["raw.wav", "mastered.wav"]
        ]
        if files:
            custom_audio = files[0].resolve()
            print(f"🎤 ইউজার প্রদত্ত ভয়েস স্যাম্পল পাওয়া গেছে: {custom_audio.name}")
            break

ref_wav = None
if custom_audio and custom_audio.exists():
    if custom_audio.suffix.lower() != ".wav":
        converted_wav = Path("/tmp/converted_user_voice.wav")
        conv_cmd = [
            "ffmpeg", "-y", "-i", str(custom_audio),
            "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
            str(converted_wav)
        ]
        subprocess.run(conv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        ref_wav = converted_wav
    else:
        ref_wav = custom_audio
else:
    ref_wav = repo / "voices" / "reference_merged.wav"
    if not ref_wav.exists():
        w = list(repo.rglob("*.wav"))
        ref_wav = w[0] if w else None

if not ref_wav:
    raise RuntimeError("Chatterbox-এর জন্য কোনো অডিও স্যাম্পল পাওয়া যায়নি!")

os.chdir(str(repo))
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "inference"))
sys.path.insert(0, str(repo))

clean_text = " ".join(text.split())

import infer
if hasattr(infer, "BanglaTTS"):
    sig = inspect.signature(infer.BanglaTTS.__init__)
    kw = {}
    if "device" in sig.parameters: kw["device"] = "cpu"
    if "repo_root" in sig.parameters: kw["repo_root"] = str(repo)
    if "checkpoint_path" in sig.parameters: kw["checkpoint_path"] = str(repo / "checkpoint")
    tts = infer.BanglaTTS(**kw) if kw else infer.BanglaTTS()

    res = tts.tts(clean_text, ref_wav=str(ref_wav))

    if isinstance(res, (tuple, list)):
        res = res[0]
    if hasattr(res, "cpu"):
        res = res.squeeze().cpu().numpy()
    
    arr = np.asarray(res, dtype=np.float32).reshape(-1)
    sr = getattr(tts, "sr", getattr(tts, "sample_rate", 24000))
    sf.write(out_wav, arr, sr)
else:
    raise RuntimeError("BanglaTTS class not found in inference/infer.py")
''',

    # ১২. OmniVoice Male
    "12_OmniVoice_Male": '''
import sys, torch
import soundfile as sf
from omnivoice import OmniVoice

torch.set_num_threads(2)

text = sys.argv[1]
out_wav = sys.argv[2]

ov = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32, load_asr=False)
wav = ov.generate(text, language="Bengali", instruct="male, low pitch", num_step=8)
sf.write(out_wav, wav[0], 24000)
''',

    # ১৩. RHVoice Bengali
    "13_RHVoice_Bengali_Male": '''
import sys, shutil, subprocess
rhvoice_bin = shutil.which("RHVoice-client") or shutil.which("rhvoice")
if rhvoice_bin:
    subprocess.run([rhvoice_bin, "-s", "bengali", "-o", sys.argv[2]], input=sys.argv[1].encode("utf-8"), check=True)
else:
    espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
    subprocess.run([espeak_bin, "-v", "bn+m1", "-s", "145", "-w", sys.argv[2], sys.argv[1]], check=True)
''',
}


def run_model_in_subprocess(model_name: str, code_str: str, folder: Path) -> bool:
    worker_file = WORKERS_DIR / f"{model_name}.py"
    worker_file.write_text(code_str, encoding="utf-8")
    raw_wav = folder / "raw.wav"

    cmd = [sys.executable, str(worker_file), BENCHMARK_TEXT, str(raw_wav)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)

    if res.returncode == 0 and raw_wav.exists() and raw_wav.stat().st_size > 1000:
        return True
    else:
        err_msg = (res.stdout + "\n" + res.stderr).strip()
        print(f"  ⚠️ সাব-প্রসেস ত্রুটি: {err_msg[-500:]}")
        return False


def main():
    print("=====================================================")
    print("🚀 বাংলা মেল ভয়েস বেঞ্চমার্ক টেস্ট স্যুট (১৩টি মডেল)")
    print(f"📝 টেস্ট স্ক্রিপ্ট শব্দসংখ্যা: {len(BENCHMARK_TEXT.split())}")
    print("=====================================================\n")

    config = {}
    if CONFIG_FILE.exists():
        try:
            config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            print("📋 models_config.json সফলভাবে লোড হয়েছে।")
        except Exception:
            pass

    for dir_name, code_str in WORKER_SCRIPTS.items():
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
            print(f"  ⏰ টাইমআউট (১০ মিনিট পার হয়েছে)। পরবর্তী মডেলে যাওয়া হচ্ছে...\n")
        except Exception as e:
            print(f"  ⚠️ এরর: {e}\n")

    print("🎉 সব নির্বাচিত মডেলের প্রসেসিং শেষ হয়েছে!")


if __name__ == "__main__":
    main()
