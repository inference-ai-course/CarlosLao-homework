# Synthetic Data Generation & Fine-Tuning (QLoRA) Assignment

A **reproducible, config-driven ML pipeline** for generating, fine-tuning, and evaluating atomic Q&A pairs from recent technical literature (via ArXiv).  
Designed for **onboarding ease**, **scalable benchmarking**, and **maintainable workflows**.

---

## Project Structure

<pre>
project-root/
├── config.py                  # Centralized configuration (paths, params, hyperparams)
├── logging_utils.py           # Centralized logging setup
├── arxiv_fetcher.py           # Fetch & group recent ArXiv papers
├── qa_generator.py            # Convert grouped summaries into fine-tuning Q&A format
├── qlora_fine_tuner.py        # Fine-tune base model with QLoRA on synthetic Q&A
├── model_evaluator.py         # Compare base vs fine-tuned model performance
├── data/                      # Input/output datasets
│   ├── synthetic_qa.jsonl     # Synthetic Q&A dataset for fine-tuning
│   └── evaluation_qa.jsonl    # Evaluation Q&A pairs
├── models/                    # Saved fine-tuned models
└── README.md
</pre>

---

## Getting Started

### Prerequisites

- Python **3.10+**
- Install dependencies:
<pre>
pip install -r requirements.txt
</pre>
- Optional Conda environment:
<pre>
conda env create -f environment.yml
</pre>

### Setup

<pre>
# Clone repo
git clone https://github.com/your-org/scientific-qa-pipeline.git
cd scientific-qa-pipeline

# Activate environment
source activate scientific-qa-pipeline

# (Optional) Create .env file to override defaults in config.py
cp .env.example .env
</pre>

---

## Pipeline Overview

| Stage               | Script                | Description                                                             |
| ------------------- | --------------------- | ----------------------------------------------------------------------- |
| **Fetch Papers**    | `arxiv_fetcher.py`    | Queries ArXiv API for recent papers, groups summaries into JSON         |
| **Generate Q&A**    | `qa_generator.py`     | Converts grouped summaries with Q&A pairs into JSONL fine-tuning format |
| **Fine-tune Model** | `qlora_fine_tuner.py` | Applies QLoRA to base model using synthetic Q&A dataset                 |
| **Evaluate Models** | `model_evaluator.py`  | Compares base vs fine-tuned model answers against reference answers     |

---

## Configuration

All parameters are centralized in **`config.py`** and can be overridden via environment variables or a `.env` file.

Example `.env`:

<pre>
ARXIV_QUERY="graph neural networks"
ARXIV_MAX_RESULTS=50
SUMMARY_GROUP_SIZE=5
BASE_MODEL_ID="unsloth/llama-3.1-8b-unsloth-bnb-4bit"
MODEL_OUTPUT_NAME="llama-3.1-8b-qlora-finetuned"
LEARNING_RATE=5e-5
BATCH_SIZE=8
NUM_EPOCHS=5
</pre>

---

## Example Usage

### 1. Fetch recent ArXiv papers

<pre>
python arxiv_fetcher.py
</pre>

### 2. Generate synthetic Q&A dataset

<pre>
python qa_generator.py
</pre>

### 3. Fine-tune model with QLoRA

<pre>
python qlora_fine_tuner.py
</pre>

### 4. Evaluate base vs fine-tuned models

<pre>
python model_evaluator.py
</pre>

---

## Evaluation Metrics

The current evaluator uses **string similarity** (`difflib.SequenceMatcher`) between generated and reference answers.  
You can extend it to include:

- **ROUGE / BLEU** for n-gram overlap
- **BERTScore** for semantic similarity
- **Embedding cosine similarity** for meaning-based scoring

---

## Utilities

- **`logging_utils.py`** — Consistent logging format across scripts
- **`config.py`** — Centralized, environment-variable-driven configuration
- **Atomic writes & pre-flight checks** — Built into scripts for robustness
