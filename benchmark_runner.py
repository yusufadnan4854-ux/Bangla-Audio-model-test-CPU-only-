name: Generate Bangla Male TTS Samples to Google Drive

on:
  workflow_dispatch:

jobs:
  run-tts-suite:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install System Tools (FFmpeg, eSpeak-NG, Git-LFS, Rclone)
        run: |
          sudo apt-get update -qq
          sudo apt-get install -y -qq ffmpeg espeak-ng git-lfs rclone build-essential python3-dev
          git lfs install --system

      - name: Install Base Python Packages
        run: |
          python -m pip install --upgrade pip
          pip install wheel setuptools cython "numpy<2.0.0"
          pip install -r requirements.txt

      - name: Install Advanced Model Dependencies (IndicVoice, Chatterbox & OmniVoice)
        run: |
          pip install loguru peft accelerate omegaconf conformer diffusers librosa pyloudnorm resemble-perth s3tokenizer pydub spacy perth || true
          pip install --no-build-isolation "pkuseg>=0.0.25" || true
          pip install --no-deps chatterbox-tts || true
          pip install git+https://github.com/Bindkushal/indic-g2p.git || true
          
          # Git LFS সহ indic-voice প্যাকেজ ক্লোন ও ইন্সটল
          rm -rf /tmp/indic-voice
          git clone https://github.com/Bindkushal/indic-voice.git /tmp/indic-voice
          cd /tmp/indic-voice && git lfs pull && pip install --no-deps . && cd -
          
          pip install git+https://github.com/k2-fsa/OmniVoice.git --no-deps || true

      - name: Configure Rclone
        shell: python
        env:
          RCLONE_CONF_DATA: ${{ secrets.RCLONE_CONF }}
        run: |
          import os
          config_path = os.path.expanduser('~/.config/rclone/rclone.conf')
          os.makedirs(os.path.dirname(config_path), exist_ok=True)
          with open(config_path, 'w', encoding='utf-8') as f:
              f.write(os.environ['RCLONE_CONF_DATA'])

      - name: Run Benchmark Generator (Male Voices)
        run: python -u benchmark_runner.py

      - name: Upload All Model Folders to Google Drive
        env:
          FOLDER_ID: ${{ secrets.GDRIVE_TTS_FOLDER_ID }}
        run: |
          echo "📤 গুগল ড্রাইভে আগের তৈরি অডিও রিপ্লেস করে নতুন অডিও আপলোড করা হচ্ছে..."
          rclone copy ./tts_results "gdrive:" --drive-root-folder-id="$FOLDER_ID" --ignore-times --drive-use-trash=false --fast-list -P
          rclone dedupe newest "gdrive:" --drive-root-folder-id="$FOLDER_ID" -q || true
          echo "✅ গুগল ড্রাইভে ওভাররাইট ও রিপ্লেস সফল!"
