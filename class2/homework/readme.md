# 🧠 Multimodal Data Extraction Pipeline with OCR, ASR & Web Scraping

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

A comprehensive pipeline for extracting and cleaning data from diverse sources—web pages, PDFs, and audio transcripts—using **Tesseract OCR**, **Web Scraping**, and **Automatic Speech Recognition (ASR)**. It includes image preprocessing, layout-aware OCR, language detection, deduplication, and PII removal to produce a high-quality textual corpus.

---

## 📦 Key Features

- ✅ Advanced image preprocessing for OCR accuracy
- ✅ Web scraping and HTML cleaning with Trafilatura
- ✅ PDF-to-text conversion via OCR
- ✅ ASR transcription from YouTube audio
- ✅ End-to-end data cleaning and deduplication

---

## 🧰 Technologies Used

| Tool               | Purpose                     |
| ------------------ | --------------------------- |
| `pytesseract`      | OCR engine                  |
| `pdf2image`        | Convert PDFs to images      |
| `trafilatura`      | Clean HTML content          |
| `yt-dlp`           | Download YouTube audio      |
| `langdetect`       | Language detection          |
| `datasketch`       | MinHash-based deduplication |
| `OpenCV`, `Pillow` | Image preprocessing         |

---

## 📌 Task Breakdown

The project is modularized into four core tasks and one shared utility script. Each task handles a distinct data source or processing step.

---

### 🧰 Main Task: Tesseract Best Practices

📄 [`task_main.py`](./task_main.py)

- **Purpose**: Centralize reusable functions for image preprocessing and text normalization.
- **Highlights**:
  - Resizing, binarization, and enhancement utilities
  - Shared text cleaning and formatting logic
- **Input**: All `.jpg`/`.png` files in [`task_main/`](./task_main/)
- **Output**: Corresponding `.txt` files in [`task_main/`](./task_main/)

---

### 🧾 Task 1: Web Scraping & HTML Cleaning

📄 [`task_bonus_1.py`](./task_bonus_1.py)

- **Purpose**: Extract abstracts from arXiv paper screenshots.
- **Highlights**:
  - OCR-based text extraction
  - Abstract isolation and cleanup
- **Input**: Latest 200 arXiv papers in subcategory `cs.CL`
- **Output**: [`arxiv_clean.json`](./arxiv_clean.json)

---

### 📄 Task 2: PDF to Text OCR

📄 [`task_bonus_2.py`](./task_bonus_2.py)

- **Purpose**: Extract full text from arXiv PDFs using OCR.
- **Highlights**:
  - PDF-to-image conversion
  - OCR-based text extraction
- **Input**: PDF links from Task 1
- **Output**: [`pdf_ocr/*.txt`](./pdf_ocr/)

---

### 🗣️ Task 3: Automatic Speech Recognition (ASR)

📄 [`task_bonus_3.py`](./task_bonus_3.py)

- **Purpose**: Transcribe academic talks using OCR and ASR.
- **Highlights**:
  - Slide text extraction via OCR
  - Spoken content transcription via ASR
  - Timestamp alignment
- **Input**: Video links in [`task_bonus_3_input.txt`](./task_bonus_3_input.txt)
- **Output**: [`talks_transcripts.jsonl`](./talks_transcripts.jsonl)

---

### 🧹 Task 4: Data Cleaning & Deduplication

📄 [`task_bonus_4.py`](./task_bonus_4.py)

- **Purpose**: Merge and refine all extracted text into a unified corpus.
- **Highlights**:
  - Duplicate removal and noise filtering
  - Token statistics for quality assessment
- **Input**: Outputs from Tasks 1–3
- **Output**:
  - [`clean_corpus.txt`](./clean_corpus.txt)
  - [`stats.md`](./stats.md)

---

## 📁 Project Structure

```plaintext
class2/
├── task_main.py                  # Main Task: Tesseract Best Practices
│   └── task_main/                # Folder for image inputs and text outputs
│       ├── *.jpg / *.png         # Input: raw images
│       └── *.txt                 # Output: preprocessed text files
│
├── task_bonus_1.py               # Task 1: Web Scraping & HTML Cleaning
│   └── arxiv_clean.json          # Output: cleaned abstract data (JSON)
│
├── task_bonus_2.py               # Task 2: PDF to Text OCR
│   └── pdf_ocr/                  # Folder for OCR-extracted text files
│       └── *.txt                 # Output: text files from PDFs
│
├── task_bonus_3.py               # Task 3: Automatic Speech Recognition (ASR)
│   ├── task_bonus_3_input.txt    # Input: YouTube video links
│   └── talks_transcripts.jsonl   # Output: transcripts with timestamps (JSONL)
│
├── task_bonus_4.py               # Task 4: Data Cleaning & Deduplication
│   ├── clean_corpus.txt          # Output: final cleaned and deduplicated corpus
│   └── stats.md                  # Output: token statistics and removal summary
│
└── README.md                     # Project documentation and usage guide
```
