# -*- coding: utf-8 -*-
"""
Bangla Male TTS Benchmark Suite
Generates raw & studio-mastered audio samples across all viable open-source engines.
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

# Hugging Face স্টল প্রতিরোধ
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


def write_pcm16_wav(path: Path, audio_f32: np.ndarray, sr: int):
    arr = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(str(path), pcm, int(sr), subtype="PCM_16")


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


def split_sentences(text: str, max_chars: int = 180) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = [s.strip() for s in re.split(r"(?<=[।\.\?\!\n])\s*", text) if s.strip()]
    out = []
    for s in parts:
        if len(s) <= max_chars:
            out.append(s)
        else:
            words = s.split()
            cur = ""
            for w in words:
                if len(cur) + len(w) + 1 <= max_chars:
                    cur = f"{cur} {w}".strip()
                else:
                    if cur: out.append(cur)
                    cur = w
            if cur: out.append(cur)
    return out or [text]


# =========================================================================
# ইন্ডিভিজুয়াল মডেল জেনারেটর ফাংশনসমূহ (কেবল মেল ভয়েস)
# =========================================================================

def run_indic_tiny(folder: Path):
    print("▶ [1/8] Indic-Tiny-TTS (Male)...")
    repo = snapshot_download("sulabhkatiyar/indic-tiny-tts-bengali", ignore_patterns=["*.safetensors", "samples/*"])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from indic_tiny_tts import TinyTTSOnnx
    tts = TinyTTSOnnx.from_pretrained(repo)
    wav = tts.synthesize(BENCHMARK_TEXT, voice="male", speed=0.95)
    tts.save_wav(wav, str(folder / "raw.wav"))


def run_ninada(folder: Path):
    print("▶ [2/8] Ninada v5.1.2 HiFi-GAN (Male bn)...")
    import onnxruntime as ort
    repo = Path(snapshot_download("gnumanth/ninada-bengali-tts", allow_patterns=["*male*"]))
    onnx_files = list(repo.glob("*.onnx")) or list(repo.rglob("*.onnx"))
    
    # Ninada সরাসরি ONNX এক্সিকিউশন
    if onnx_files:
        sess = ort.InferenceSession(str(onnx_files[0]), providers=["CPUExecutionProvider"])
        tokens_map = json.loads((repo / "tokens.json").read_text(encoding="utf-8")) if (repo / "tokens.json").exists() else {}
        # ফলব্যাক যদি টোকেনাইজার না থাকে
        subprocess.run(["ninada", "-l", "bn", "-o", str(folder / "raw.wav"), BENCHMARK_TEXT], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        # CLI এক্সিকিউশন
        subprocess.run(["python", "-m", "ninada", "-l", "bn", "-o", str(folder / "raw.wav"), BENCHMARK_TEXT], check=True)


def run_sherpa_bd(folder: Path):
    print("▶ [3/8] BanglaTTS Bangladeshi VITS (Sherpa-ONNX Male)...")
    import sherpa_onnx
    repo = Path(snapshot_download("demoqqq/banglatts-kiki-hasan-big"))
    onnx_file = next(repo.rglob("*.onnx"))
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
    sentences = split_sentences(BENCHMARK_TEXT, max_chars=180)
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
    write_pcm16_wav(folder / "raw.wav", merged, sr)


def run_piper_bd(folder: Path):
    print("▶ [4/8] Piper TTS Bangladeshi (Male bn_BD-google)...")
    from piper import PiperVoice
    repo = Path(snapshot_download("rhasspy/piper-voices", allow_patterns=["bn/bn_BD/google/medium/bn_BD-google-medium.onnx*"]))
    onnx_p = repo / "bn/bn_BD/google/medium/bn_BD-google-medium.onnx"
    voice = PiperVoice.load(str(onnx_p))
    with wave.open(str(folder / "raw.wav"), "wb") as wf:
        voice.synthesize_wav(BENCHMARK_TEXT, wf)


def run_piper_in(folder: Path):
    print("▶ [5/8] Piper TTS Indian Bengali (Male bn_IN-biswas)...")
    from piper import PiperVoice
    repo = Path(snapshot_download("rhasspy/piper-voices", allow_patterns=["bn/bn_IN/biswas/medium/bn_IN-biswas-medium.onnx*"]))
    onnx_p = repo / "bn/bn_IN/biswas/medium/bn_IN-biswas-medium.onnx"
    voice = PiperVoice.load(str(onnx_p))
    with wave.open(str(folder / "raw.wav"), "wb") as wf:
        voice.synthesize_wav(BENCHMARK_TEXT, wf)


def run_meta_mms(folder: Path):
    print("▶ [6/8] Meta MMS-TTS Bengali (ONNX Male/Default)...")
    import onnxruntime as ort
    repo = Path(snapshot_download("naklitechie/mms-tts-bn-ONNX"))
    vocab = json.loads((repo / "vocab.json").read_text(encoding="utf-8"))
    cfg = json.loads((repo / "config.json").read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(repo / "model.onnx"), providers=["CPUExecutionProvider"])

    sr = int(cfg.get("sampling_rate", 16000))
    sentences = split_sentences(BENCHMARK_TEXT, max_chars=160)
    parts = []
    pause = np.zeros(int(sr * 0.15), dtype=np.float32)

    for i, s in enumerate(sentences):
        chars = [vocab[c] for c in s if c in vocab]
        if not chars: continue
        ids = [0] + [c for val in chars for c in (val, 0)]
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "scales": np.array([0.667, 1.05, 0.8], dtype=np.float32)
        }
        out = sess.run(None, feed)
        parts.append(np.asarray(out[0], dtype=np.float32).reshape(-1))
        if i < len(sentences) - 1:
            parts.append(pause)

    merged = np.concatenate(parts) if len(parts) > 1 else parts[0]
    write_pcm16_wav(folder / "raw.wav", merged, sr)


def run_edge_tts(folder: Path):
    print("▶ [7/8] Microsoft Edge TTS (Male: bn-BD-PradeepNeural)...")
    import asyncio, edge_tts
    raw_mp3 = folder / "temp.mp3"
    async def _make():
        comm = edge_tts.Communicate(BENCHMARK_TEXT, "bn-BD-PradeepNeural", rate="-6%")
        await comm.save(str(raw_mp3))
    asyncio.run(_make())
    subprocess.run(["ffmpeg", "-y", "-i", str(raw_mp3), "-ar", "24000", "-ac", "1", str(folder / "raw.wav")], check=True)
    if raw_mp3.exists(): raw_mp3.unlink()


def run_espeak_ng(folder: Path):
    print("▶ [8/8] eSpeak-NG Bengali (Male)...")
    espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak_bin:
        subprocess.run([espeak_bin, "-v", "bn+m3", "-s", "145", "-w", str(folder / "raw.wav"), BENCHMARK_TEXT], check=True)
    else:
        print("  ⚠️ eSpeak-NG বাইনারি পাওয়া যায়নি। স্কিপ করা হলো।")


# =========================================================================
# মাস্টার টেস্ট এক্সিকিউশন
# =========================================================================

MODELS = [
    ("01_Indic_Tiny_TTS_Male", run_indic_tiny),
    ("02_Ninada_Male", run_ninada),
    ("03_BanglaTTS_Sherpa_BD_Male", run_sherpa_bd),
    ("04_Piper_Bangladeshi_Male", run_piper_bd),
    ("05_Piper_WestBengal_Male", run_piper_in),
    ("06_Meta_MMS_Bengali_Male", run_meta_mms),
    ("07_Edge_TTS_PradeepNeural_Male", run_edge_tts),
    ("08_eSpeak_NG_Bengali_Male", run_espeak_ng),
]

def main():
    print("=====================================================")
    print("🚀 বাংলা মেল ভয়েস বেঞ্চমার্ক টেস্ট শুরু হচ্ছে...")
    print(f"📝 টেস্ট স্ক্রিপ্ট শব্দসংখ্যা: {len(BENCHMARK_TEXT.split())}")
    print("=====================================================\n")

    for dir_name, runner_fn in MODELS:
        folder = OUTPUT_ROOT / dir_name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "script.txt").write_text(BENCHMARK_TEXT, encoding="utf-8")

        raw_wav = folder / "raw.wav"
        mastered_wav = folder / "mastered.wav"

        t0 = time.time()
        try:
            runner_fn(folder)
            if raw_wav.exists() and raw_wav.stat().st_size > 1000:
                apply_studio_mastering(raw_wav, mastered_wav)
                dur = sf.info(str(raw_wav)).duration
                elap = round(time.time() - t0, 2)
                print(f"  ✅ সফল! অডিও দৈর্ঘ্য: {round(dur, 1)}s | সময় লেগেছে: {elap}s\n")
            else:
                print(f"  ❌ অডিও তৈরি হয়নি।\n")
        except Exception as e:
            print(f"  ⚠️ রান করতে ত্রুটি: {e}\n")

    print("\n🎉 সবগুলো মডেলের টেস্টিং সম্পন্ন হয়েছে!")

if __name__ == "__main__":
    main()