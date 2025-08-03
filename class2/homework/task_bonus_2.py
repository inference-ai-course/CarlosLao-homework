import os
import sys
import shutil
import requests
import arxiv
import pytesseract
from pdf2image import convert_from_bytes

def check_poppler():
    """Ensure Poppler is installed and available in system PATH."""
    if shutil.which("pdftoppm") is None:
        print("Poppler is not installed or not in your system PATH.")
        print("Please install Poppler to enable PDF conversion.")
        sys.exit(1)

def pdf_url_to_images(url, dpi=300):
    """Download a PDF from a URL and convert it to images."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return convert_from_bytes(response.content, dpi=dpi)
    except Exception as e:
        print(f"Failed to convert PDF from URL: {url}\n    Error: {e}")
        return []

def batch_ocr(output_folder="pdf_ocr", layout=True, max_results=5):
    """Perform OCR on a batch of arXiv PDFs and save results as text files."""
    check_poppler()
    
    # Prepare output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_folder)
    os.makedirs(output_path, exist_ok=True)

    # Configure OCR layout mode
    ocr_config = '--psm 1' if layout else '--psm 3'

    # Initialize arXiv client and search
    client = arxiv.Client()
    search = arxiv.Search(
        query="cs.CL",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    print(f"Starting OCR for {max_results} arXiv PDFs...\n")

    for result in client.results(search):
        url = result.pdf_url
        print(f"Processing: {url}")

        images = pdf_url_to_images(url)
        if not images:
            print("Skipping due to conversion failure.\n")
            continue

        full_text = ''
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img, config=ocr_config)
            full_text += f"\n--- Page {i+1} ---\n{page_text}"

        # Generate output filename
        base_name = os.path.basename(url).replace('.pdf', '')
        txt_filename = f"{base_name}.txt"
        file_path = os.path.join(output_path, txt_filename)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"OCR saved: {txt_filename}\n")
        except Exception as e:
            print(f"Error saving {txt_filename}: {e}\n")

    print("Batch OCR completed.")

if __name__ == "__main__":
    # Customize these settings
    output_folder = "pdf_ocr"     # Folder to save OCR .txt files
    retain_layout = True          # Set to False to disable layout-aware OCR
    batch_ocr(output_folder=output_folder, layout=retain_layout, max_results=5)