# RAG Pipeline for Semantic Search on arXiv Papers

This project implements a full Retrieval-Augmented Generation (RAG) pipeline using arXiv papers from the `cs.CL` category. It includes:

- PDF download from arXiv  
- Text extraction using PyMuPDF  
- Chunking into token-sized segments  
- Embedding with Sentence Transformers  
- FAISS indexing for fast semantic search  
- FastAPI service for query-based retrieval and reporting  
- Jupyter Notebook demo with interactive search  

---

## Project Structure

<pre>
.
├── rag_pipeline.py             # Full pipeline script
├── main.py                     # FastAPI app
├── semantic_search_demo.ipynb  # Jupyter notebook demo
├── arxiv_pdfs/                 # Downloaded PDFs
├── arxiv_texts/                # Extracted text files
├── arxiv_chunks/               # Chunked text segments
├── faiss_index.bin             # FAISS index file
├── chunk_metadata.pkl          # List of chunk filenames
├── retrieval_log.json          # Saved queries and results
├── retrieval_report.md         # Generated report
└── README.md                   # This file
</pre>

---

## Setup Instructions

### 1. Clone the repository

<pre>
git clone https://github.com/your-username/rag-arxiv-search.git
cd rag-arxiv-search
</pre>

### 2. Create a virtual environment (optional but recommended)

<pre>
python3 -m venv venv
source venv/bin/activate
</pre>

### 3. Install dependencies

<pre>
pip install -r requirements.txt
</pre>

If you don’t have a `requirements.txt`, install manually:

<pre>
pip install arxiv faiss-cpu fitz tqdm sentence-transformers fastapi uvicorn ipywidgets
</pre>

---

## Run the RAG Pipeline

This script downloads papers, extracts text, chunks it, embeds it, and builds a FAISS index.

<pre>
python rag_pipeline.py
</pre>

---

## Run the FastAPI Service

After building the index, launch the API:

<pre>
python main.py
</pre>

Or use `uvicorn` directly:

<pre>
uvicorn main:app --reload
</pre>

---

## FastAPI Endpoints

### `/search`
Returns top-3 matching chunks for a given query.

Example:

<pre>
GET /search?q=transformer models
</pre>

### `/report`
Generates a Markdown-style retrieval report from the top queries in `retrieval_log.json`.

Example:

<pre>
GET /report
GET /report?max_queries=5
</pre>

### `/report/download`
Returns the report as a downloadable Markdown file.

Example:

<pre>
GET /report/download
</pre>

---

## Use the Jupyter Notebook

Launch Jupyter and open `semantic_search_demo.ipynb`:

<pre>
jupyter notebook
</pre>

- Type a query  
- Click Submit  
- View top-3 matching chunks  
- All queries are saved to `retrieval_log.json`  
- You can generate a report using the `/report` or `/report/download` endpoints

---

## Retrieval Report

See `retrieval_log.json` for saved queries and results. You can also generate a Markdown report manually:

<pre>
import json
with open("retrieval_log.json") as f:
    log = json.load(f)
for entry in log[:5]:
    print(entry["query"])
    for r in entry["results"]:
        print("-", r["chunk_id"])
</pre>

Or download it via:

<pre>
GET /report/download
</pre>

---

## Optional Enhancements

- Add metadata (title, abstract) to chunks  
- Convert chunks to JSONL for integration  
- Deploy FastAPI with Docker or Hugging Face Spaces  
- Add frontend UI with Streamlit or Gradio  

---

## License

MIT License © 2025 Carlos Lao

---

## Questions?

Feel free to open an issue or reach out.
