import os
import re
import glob
import json
import utils
from langdetect import detect
from datasketch import MinHash, MinHashLSH

# ---------- Configuration ----------
TASK_1_OUTPUT = "arxiv_clean.json"
TASK_2_OUTPUT = "pdf_ocr/*.txt"
TASK_3_OUTPUT = "talks_transcripts.jsonl"
OUTPUT_CORPUS = "clean_corpus.txt"
OUTPUT_STATS = "stats.md"
MINHASH_THRESHOLD = 0.7
NUM_PERMUTATIONS = 128


# ---------- Utility Functions ----------
def extract_text():
    """Extracts raw text from multiple sources."""
    all_texts = []

    # Task 1: JSON abstracts
    task_1_path = utils.get_path(TASK_1_OUTPUT)
    if os.path.exists(task_1_path):
        with open(task_1_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_texts.extend([entry.get("abstract", "") for entry in data])

    # Task 2: OCR .txt files
    for file_path in glob.glob(utils.get_path(TASK_2_OUTPUT)):
        with open(file_path, "r", encoding="utf-8") as f:
            all_texts.append(f.read())

    # Task 3: JSONL transcripts
    task_3_path = utils.get_path(TASK_3_OUTPUT)
    if os.path.exists(task_3_path):
        with open(task_3_path, "r", encoding="utf-8") as f:
            all_texts.extend([json.loads(line).get("text", "") for line in f])

    return all_texts


def clean_text(text):
    """Cleans text by removing HTML, PII, and repetitive patterns."""
    text = re.sub(r"<[^>]+>", "", text)  # HTML tags
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "", text)  # Emails
    text = re.sub(r"\b(?:\d{4}[- ]?){3}\d{4}\b", "", text)  # Credit cards
    text = re.sub(r"\b\d{10,}\b", "", text)  # Phone numbers
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)  # Repetitive n-grams
    return text.strip()


def is_english(text):
    """Detects if the text is in English."""
    try:
        return detect(text) == "en"
    except:
        return False


# ---------- Main Cleaning Pipeline ----------
def clean_and_deduplicate():
    all_texts = extract_text()
    total_raw = len(all_texts)
    cleaned_texts = []
    minhashes = []
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=NUM_PERMUTATIONS)

    for i, raw_text in enumerate(all_texts):
        if not raw_text or not is_english(raw_text):
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        m = MinHash(num_perm=NUM_PERMUTATIONS)
        for token in cleaned.split():
            m.update(token.encode("utf8"))

        # Check for near-duplicates
        if not any(m.jaccard(existing) > MINHASH_THRESHOLD for existing in minhashes):
            minhashes.append(m)
            lsh.insert(f"doc_{i}", m)
            cleaned_texts.append(cleaned)

    # Save cleaned corpus
    corpus_path = utils.get_path(OUTPUT_CORPUS)
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(cleaned_texts))

    # Save stats
    total_cleaned = len(cleaned_texts)
    total_tokens = sum(len(text.split()) for text in cleaned_texts)
    total_removed = total_raw - total_cleaned
    removal_pct = (total_removed / total_raw) * 100 if total_raw else 0

    stats_path = utils.get_path(OUTPUT_STATS)
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"# Corpus Cleaning Stats\n\n")
        f.write(f"- **Total documents before cleaning:** {total_raw}\n")
        f.write(f"- **Total documents after cleaning:** {total_cleaned}\n")
        f.write(f"- **Total tokens:** {total_tokens}\n")
        f.write(f"- **Duplicates removed:** {total_removed}\n")
        f.write(f"- **Removal percentage:** {removal_pct:.2f}%\n")

    print(f"Cleaning complete. Saved to '{OUTPUT_CORPUS}' and '{OUTPUT_STATS}'.")


# ---------- Run ----------
if __name__ == "__main__":
    clean_and_deduplicate()
