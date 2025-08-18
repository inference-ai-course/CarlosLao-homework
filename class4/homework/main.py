import json
import logging
import os
import pickle
from pathlib import Path

import faiss
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from sentence_transformers import SentenceTransformer

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
LOG_PATH = os.path.join(BASE_DIR, "retrieval_log.json")

# === FASTAPI SERVICE ===
app = FastAPI()

# Load model, index, and metadata only if they exist
if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        chunk_ids = pickle.load(f)
else:
    logger.warning("FAISS index or metadata not found. Run the script to generate them.")



def generate_retrieval_report(log_path=LOG_PATH, max_queries=5, save_path=None):
    path = Path(log_path)
    if not path.exists():
        content = "# Retrieval Report\n\nNo queries found."
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
        return content

    with open(path, "r", encoding="utf-8") as f:
        log = json.load(f)

    top_queries = log[:max_queries]
    report_lines = ["# Retrieval Report\n"]

    for entry in top_queries:
        query = entry["query"]
        report_lines.append(f"## Query: {query}\n")
        for i, result in enumerate(entry["results"], 1):
            chunk_id = result["chunk_id"]
            text = result["text"].strip().replace("\n", " ")
            snippet = text[:300] + "..." if len(text) > 300 else text
            report_lines.append(f"**Result {i}** (`{chunk_id}`):\n")
            report_lines.append(f"> {snippet}\n")
        report_lines.append("\n")

    markdown = "\n".join(report_lines)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(markdown)

    return markdown

@app.get("/report/download")
def download_report(max_queries: int = 5):
    report_path = os.path.join(BASE_DIR, "retrieval_report.md")
    generate_retrieval_report(max_queries=max_queries, save_path=report_path)

    if not os.path.exists(report_path):
        return JSONResponse(content={"error": "Report not found"}, status_code=500)

    return FileResponse(
        path=report_path,
        media_type="text/markdown",
        filename="retrieval_report.md"
    )

@app.get("/search")
def search(q: str = Query(..., description="Search query"), k: int = 3):
    try:
        query_embedding = model.encode([q]).astype("float32")
        distances, indices = index.search(query_embedding, k)

        results = []
        for i in indices[0]:
            chunk_file = os.path.join(CHUNK_DIR, chunk_ids[i])
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk_text = f.read().strip()
            results.append({
                "chunk_id": chunk_ids[i],
                "text": chunk_text,
                "score": float(distances[0][list(indices[0]).index(i)])
            })

        return {"query": q, "results": results}

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.get("/report")
def report(max_queries: int = 5):
    markdown = generate_retrieval_report(max_queries=max_queries)
    return JSONResponse(content={"report": markdown})


# === MAIN EXECUTION ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("All tasks completed.")
