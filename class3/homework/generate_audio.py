import os
from typing import List
from gtts import gTTS
import utils

def read_prompts(file_path: str) -> List[str]:
    """
    Reads non-empty, stripped lines from a text file.

    Parameters:
        file_path (str): Path to the input text file.

    Returns:
        List[str]: List of prompt strings.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return []

def clean_audio_dir(subfolder: str) -> None:
    """
    Deletes all files in the given subdirectory inside the script's folder.

    Parameters:
        subfolder (str): Name of the directory to clean.
    """
    audio_dir = utils.get_script_dir(subfolder)
    for entry in os.listdir(audio_dir):
        path = utils.get_file_path(entry, subfolder)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
        except Exception as e:
            print(f"Could not delete '{path}': {e}")

def text_to_mp3(text: str, output_folder: str, filename: str) -> None:
    """
    Converts text to speech and saves as an MP3 file.

    Parameters:
        text (str): The input text to synthesize.
        output_folder (str): Target folder where audio will be saved.
        filename (str): Name of the output MP3 file.
    """
    output_path = utils.get_file_path(filename, output_folder)
    try:
        tts = gTTS(text)
        tts.save(output_path)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error generating '{filename}': {e}")

if __name__ == "__main__":
    input_filename = "prompt.txt"
    audio_folder = "audio"

    # Clean output directory before generating files
    clean_audio_dir(audio_folder)

    # Read and process prompt lines
    prompts = read_prompts(utils.get_file_path(input_filename))
    for i, prompt in enumerate(prompts, start=1):
        filename = f"sample{i if i > 1 else ''}.mp3"
        text_to_mp3(prompt, audio_folder, filename)
