import os
import subprocess
from pathlib import Path

folder = Path("audio_speech/CREMA_D")

for mp3_file in folder.glob("*.mp3"):
    wav_file = mp3_file.with_suffix(".wav")
    subprocess.run([
        "ffmpeg",
        "-i", str(mp3_file),
        str(wav_file)
    ], check=True)

print("✅ Conversion complete")

for mp3_file in folder.glob("*.mp3"):
    mp3_file.unlink()

print("🧹 All MP3 files deleted.")