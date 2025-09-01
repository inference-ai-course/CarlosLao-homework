import requests
import utils
import os

# Endpoint URL
url = "http://localhost:8000/chat/"

# Audio folder setup
audio_folder = "audio"
audio_dir = utils.get_script_dir(audio_folder)
audio_extensions = (".mp3", ".wav", ".flac", ".aac", ".ogg")

# Collect audio files
audio_files = [
    utils.get_file_path(f, audio_folder)
    for f in os.listdir(audio_dir)
    if f.lower().endswith(audio_extensions)
]

# Send each file to the endpoint
for file_path in audio_files:
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/mpeg")}
            response = requests.post(url, files=files)
            response.raise_for_status()
            print(f"{file_path} → {response.json()}")
    except Exception as e:
        print(f"Error with {file_path}: {e}")

print(f"\nFinished processing {len(audio_files)} audio files.")