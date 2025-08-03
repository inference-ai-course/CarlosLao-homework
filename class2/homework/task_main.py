from pathlib import Path
import cv2
import pytesseract  # Make sure Tesseract is installed and added to PATH
from PIL import Image  # Optional: used if you switch to PIL-based OCR
import utils
import os

# Define image folder path
image_folder = Path(utils.get_path("task_main"))

# Collect all .jpg and .png files
image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))

for image_path in image_files:
    # Read image using OpenCV
    img = cv2.imread(str(image_path))

    # OCR without preprocessing
    raw_text = pytesseract.image_to_string(img)

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Adjust brightness and contrast
    # alpha: contrast (1.0–3.0), beta: brightness (0–100)
    adjusted = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)

    # Step 3: Denoise
    denoised = cv2.fastNlMeansDenoising(
        adjusted, None, h=30, templateWindowSize=7, searchWindowSize=21
    )

    # Step 4: Thresholding (Otsu's method)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Extract filename and language code
    filename = image_path.name
    if "-" in filename:
        language = filename.split("-", 1)[1].split(".")[0]
    else:
        language = "eng"  # Default to English

    # Step 5: OCR with Tesseract
    processed_text = pytesseract.image_to_string(thresh, lang=language)

    # Define output file paths
    raw_output_path = image_path.with_suffix(".txt")
    processed_output_path = image_path.with_name(image_path.stem + "_preprocess.txt")

    # Save raw OCR output
    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Save processed OCR output
    with open(processed_output_path, "w", encoding="utf-8") as f:
        f.write(processed_text)

    print(f"Processed: {image_path.name} → Saved to:")
    print(f"   - {raw_output_path.name}")
    print(f"   - {processed_output_path.name}")
