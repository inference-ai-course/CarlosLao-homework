import logging
import os
import pickle

import arxiv
import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === LOGGING CONFIGURATION ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
CATEGORY = "cs.CL"
MAX_RESULTS = 50
CHUNK_SIZE = 512
OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === BASE DIRECTORY ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "arxiv_pdfs")
TEXT_DIR = os.path.join(BASE_DIR, "arxiv_texts")
CHUNK_DIR = os.path.join(BASE_DIR, "arxiv_chunks")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(BASE_DIR, "chunk_metadata.pkl")

# === SETUP DIRECTORIES ===
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

# === CLEANUP FUNCTION ===
def clear_directory(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        logger.info(f"Cleared directory: {path}")

# === DOWNLOAD PDFs ===
def download_arxiv_pdfs(category, max_results):
    logger.info(f"Downloading {max_results} PDFs from arXiv category: {category}")
    search = arxiv.Search(
        query=category,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    for result in tqdm(search.results(), total=max_results):
        pdf_url = result.pdf_url
        paper_id = result.get_short_id()
        pdf_path = os.path.join(PDF_DIR, f"{paper_id}.pdf")
        if not os.path.exists(pdf_path):
            result.download_pdf(filename=pdf_path)
            logger.debug(f"Downloaded: {pdf_path}")

# === EXTRACT TEXT FROM PDFs ===
def extract_text_from_pdfs():
    logger.info(f"Extracting text from PDFs in '{PDF_DIR}'...")
    for filename in tqdm(os.listdir(PDF_DIR)):
        if not filename.endswith(".pdf"):
            continue
        pdf_path = os.path.join(PDF_DIR, filename)
        text_path = os.path.join(TEXT_DIR, filename.replace(".pdf", ".txt"))

        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                page_text = page.get_text()
                full_text += page_text.strip() + "\n"
            doc.close()

            with open(text_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            logger.debug(f"Extracted text to: {text_path}")
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

# === CHUNK TEXT ===
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    return chunks

# === CHUNK ALL TEXT FILES ===
def chunk_all_texts():
    logger.info(f"Chunking texts from '{TEXT_DIR}' into '{CHUNK_DIR}'...")
    for filename in tqdm(os.listdir(TEXT_DIR)):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(TEXT_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        base_name = filename.replace(".txt", "")
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{base_name}_chunk{i+1}.txt"
            chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
            with open(chunk_path, "w", encoding="utf-8") as cf:
                cf.write(chunk)
        logger.debug(f"{base_name}: {len(chunks)} chunks created.")

# === EMBED AND INDEX CHUNKS ===
def build_faiss_index():
    logger.info(f"Embedding and indexing chunks from '{CHUNK_DIR}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    chunk_texts = []
    chunk_ids = []

    for filename in tqdm(sorted(os.listdir(CHUNK_DIR))):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(CHUNK_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                chunk_texts.append(text)
                chunk_ids.append(filename)

    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunk_ids, f)

    logger.info(f"FAISS index built with {len(chunk_texts)} chunks.")
    logger.info(f"Index saved to: {INDEX_PATH}")
    logger.info(f"Metadata saved to: {METADATA_PATH}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    clear_directory(PDF_DIR)
    clear_directory(TEXT_DIR)
    clear_directory(CHUNK_DIR)

    download_arxiv_pdfs(CATEGORY, MAX_RESULTS)
    extract_text_from_pdfs()
    chunk_all_texts()
    build_faiss_index()

    logger.info("All tasks completed.")
