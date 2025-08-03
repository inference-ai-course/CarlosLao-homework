import os
import re
import tempfile
import json
import utils
import whisper
from yt_dlp import YoutubeDL
from yt_dlp.utils import PostProcessingError, DownloadError


def get_video_urls(video_source_urls_input):
    """
    Extracts all YouTube video URLs from a given input file.

    Args:
        video_source_urls_input (str): Path to the input file containing URLs or text.

    Returns:
        List[str]: A list of valid YouTube video URLs.
    """
    path = utils.get_path(video_source_urls_input)
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
    return re.findall(r"https://www\.youtube\.com/watch\?v=[\w-]+(?:&[\w=]+)*", content)


def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.

    Args:
        url (str): YouTube video URL.

    Returns:
        str: Video ID or 'unknown_id' if not found.
    """
    match = re.search(r"v=([\w-]+)", url)
    return match.group(1) if match else "unknown_id"


def save_transcript_jsonl(video_id, segments, output_file):
    """
    Saves transcription segments to a JSONL file.

    Args:
        video_id (str): Unique video identifier.
        segments (List[Dict]): Transcription segments from Whisper.
        output_file (str): Path to the output JSONL file.
    """
    with open(output_file, mode="a", encoding="utf-8") as writer:
        for seg in segments:
            writer.write(
                json.dumps(
                    {
                        "id": video_id,
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip(),
                    }
                )
                + "\n"
            )


def transcribe_youtube(url, model_size="base"):
    """
    Downloads audio from a YouTube video and transcribes it using Whisper.

    Args:
        url (str): YouTube video URL.
        model_size (str): Whisper model size (e.g., 'base', 'small', 'medium', 'large').

    Returns:
        List[Dict]: Transcription segments or empty list on failure.
    """
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(temp_fd)

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": temp_path.replace(".mp3", "") + ".%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_file = (
                ydl.prepare_filename(info)
                .replace(".webm", ".mp3")
                .replace(".m4a", ".mp3")
            )

        model = whisper.load_model(model_size)
        result = model.transcribe(audio_file)
        os.remove(audio_file)  # Clean up temporary audio file
        return result["segments"]

    except (PostProcessingError, DownloadError) as e:
        print(f"Error processing {url}: {e}")
        return []


def process_automatic_speech_recognition(input_file_path, output_file_path):
    """
    Main pipeline to extract YouTube URLs, transcribe audio, and save transcripts.

    Args:
        input_file_path (str): Path to input file containing YouTube URLs.
        output_file_path (str): Path to output JSONL file for transcripts.
    """
    urls = get_video_urls(input_file_path)
    output_file = utils.get_path(output_file_path)

    # Clear output file before writing
    open(output_file, "w").close()

    for url in urls:
        print(f"Processing: {url}")
        video_id = extract_video_id(url)
        segments = transcribe_youtube(url)
        save_transcript_jsonl(video_id, segments, output_file)
        print(f"Saved {len(segments)} segments for {video_id}")


# ----------- RUN SCRIPT -----------
if __name__ == "__main__":
    input_file = "task_bonus_3_input.txt"
    output_file = "talks_transcripts.jsonl"
    process_automatic_speech_recognition(input_file, output_file)
