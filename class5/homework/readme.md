# Hybrid Search Engine for NLP Documents

This project implements a hybrid search engine that combines semantic search (via FAISS and Sentence Transformers) with keyword search (via SQLite FTS5 and BM25). It allows fast and accurate retrieval of document chunks based on natural language queries.

## Features

- Semantic search using FAISS and transformer embeddings  
- Keyword search using SQLite FTS5 or BM25  
- Hybrid search with Reciprocal Rank Fusion  
- FastAPI endpoint for querying the index  
- Evaluation notebook with Recall@3 metrics  
- Modular design with structured logging and documentation  

## Directory Structure

<pre>
class5/homework/
├── app/
│   ├── config.py          # Centralized paths and logger
│   ├── indexing.py        # Index builder (SQLite + FAISS)
│   ├── retrieval.py       # Search logic (semantic, keyword, hybrid)
│   ├── api.py             # FastAPI endpoint
├── data/                  # Stores index files and database
├── notebooks/
│   └── evaluation.ipynb   # Performance testing
├── run_indexing.py        # Script to build index from sample data
├── main.py                # Launches FastAPI server
├── README.md
└── requirements.txt
</pre>

## Setup

1. **Clone the repository**

<pre>
git clone https://github.com/your-username/hybrid-search-nlp.git
cd hybrid-search-nlp/class5/homework
</pre>

2. **Create a virtual environment**

<pre>
python -m venv vllm-env
source vllm-env/bin/activate  # or vllm-env\Scripts\activate on Windows
</pre>

3. **Install dependencies**

<pre>
pip install -r requirements.txt
</pre>

4. **Build the index**

<pre>
python run_indexing.py
</pre>

5. **Launch the API**

<pre>
python main.py
</pre>

6. **Query the endpoint**

Visit: `http://localhost:8000/hybrid_search?q=transformers`

## Evaluation

Open the notebook:

<pre>
jupyter notebook notebooks/evaluation.ipynb
</pre>

Run all cells to compare semantic, keyword, and hybrid search performance using Recall@3.

## Author

Carlos Lao — refactored for clarity, documentation, and structured logging style.
