# Scientific Summarization & Reward Modeling Pipeline

A reproducible ML pipeline for fetching scientific papers, generating faithful and accessible summaries, training a reward model, and evaluating outputs. Built for onboarding ease, config‑driven workflows, and scalable benchmarking.

---

## Project Structure

<pre>
class8/homework/
├── config.py                 # Centralized configuration
├── logging_utils.py          # Logging setup and helpers
├── arxiv_fetcher.py          # Fetch and download papers from arXiv
├── paper_summarizer.py       # Summarize PDFs into chosen/rejected pairs
├── reward_model_trainer.py   # Train reward model on summaries
├── evaluation_runner.py      # Evaluate summaries with metrics + reward model
├── homework.ipynb            # Notebook linking all scripts together
├── data/                     # Input/output datasets (PDFs, JSONL, results)
└── README.md
</pre>

---

## Getting Started

### Prerequisites

- Python 3.10+
- <pre>pip install -r requirements.txt</pre>
- Optional: <pre>conda env create -f environment.yml</pre>

### Setup

<pre>
# Clone repo
git clone https://github.com/your-org/scientific-summarization-pipeline.git
cd class8/homework

# Activate environment
source activate sci-sum-pipeline

# (Optional) Run pre-flight checks
python preflight.py
</pre>

---

## Pipeline Overview

| Stage           | Script                    | Description                                               |
| --------------- | ------------------------- | --------------------------------------------------------- |
| Ingestion       | `arxiv_fetcher.py`        | Query arXiv and download PDFs into the raw data folder    |
| Summarization   | `paper_summarizer.py`     | Chunk PDFs and generate chosen/rejected summaries via LLM |
| Reward Training | `reward_model_trainer.py` | Train a reward model on paired summaries                  |
| Evaluation      | `evaluation_runner.py`    | Score summaries with ROUGE, BERTScore, and reward model   |
| Integration     | `homework.ipynb`          | End‑to‑end notebook linking all scripts together          |

---

## Sample Config

All configuration is centralized in `config.py` and can be overridden via environment variables:

<pre>
# Example .env
LLAMA_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
MAX_NEW_TOKENS=150
BATCH_SIZE=2
ARXIV_QUERY="machine learning"
MAX_RESULTS=10
REWARD_MODEL_NAME=microsoft/deberta-v3-base
REWARD_MAX_LENGTH=1024
REWARD_BATCH_SIZE=8
REWARD_LEARNING_RATE=5e-6
</pre>

---

## Evaluation Metrics

- **ROUGE**: Overlap with reference summaries
- **BERTScore**: Semantic similarity to references
- **Reward Score**: Learned preference model scoring
- **Top‑k Analysis**: Identify best summaries by metric vs reward model

---

## Utilities

- `logging_utils.py`: Centralized logging configuration
- `config.py`: Centralized environment and hyperparameter configs
