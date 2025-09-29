# paper_summarizer.py
"""Summarizes downloaded PDFs into chosen/rejected pairs for reward modeling.

This script loads PDFs from the raw folder, extracts text, chunks it, and uses a
language model to produce a faithful (chosen) and accessible (rejected) summary.
Outputs are written as JSONL entries to the configured reward data file.
"""

import gc
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from config import Config
from logging_utils import configure_logging
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger(__name__)


class PaperSummarizer:
    """Generate chosen/rejected summaries from PDF content using an LLM."""

    def __init__(self) -> None:
        """Initialize model and tokenizer with configured quantization and dtype."""
        quant_config = self._quant_config()
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.LLAMA_MODEL_NAME,
            device_map=Config.DEVICE,
            quantization_config=quant_config,
            token=Config.HUGGINGFACE_HUB_TOKEN,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.LLAMA_MODEL_NAME,
            token=Config.HUGGINGFACE_HUB_TOKEN,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def _quant_config(self) -> Optional[BitsAndBytesConfig]:
        """Return BitsAndBytes quantization config according to settings.

        Returns:
            Optional[BitsAndBytesConfig]: Config object if quantization is enabled; None otherwise.
        """
        dtype = torch.float16 if Config.COMPUTE_DTYPE == "float16" else torch.bfloat16
        if Config.QUANT_MODE == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=dtype)
        if Config.QUANT_MODE == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        return None

    def _sanitize_text(self, text: str) -> str:
        """Clean control characters and ensure UTF-8 text.

        Args:
            text (str): Input text.

        Returns:
            str: Sanitized text string.
        """
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("\x00", " ")
        text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
        text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        return text

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the current tokenizer.

        Args:
            text (str): Input text.

        Returns:
            int: Number of tokens.
        """
        if not text:
            return 0
        text = self._sanitize_text(text)
        try:
            enc = self.tokenizer([text], add_special_tokens=False)
            return len(enc["input_ids"][0])
        except Exception as e:
            log.error(
                "Tokenizer failed on text (first 200 chars): %r | %s", text[:200], e
            )
            return 0

    def _format_prompt(self, system_prompt: str, user_text: str) -> str:
        """Format a chat-style prompt for generation.

        Args:
            system_prompt (str): System role content.
            user_text (str): User role content.

        Returns:
            str: Rendered prompt string ready for tokenization/generation.
        """
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(
        self, prompts: List[str], temperature: float, top_k: int, top_p: float
    ) -> List[str]:
        """Generate completions for a list of prompts.

        Args:
            prompts (List[str]): Prepared prompt strings.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling parameter.
            top_p (float): Nucleus sampling parameter.

        Returns:
            List[str]: Generated texts aligned with input prompts.
        """
        clean_prompts: List[str] = []
        for p in prompts:
            if p is None:
                continue
            try:
                s = self._sanitize_text(str(p)).strip()
            except Exception as e:
                log.warning("Skipping non-string prompt %r (%s)", p, e)
                continue
            if s:
                clean_prompts.append(s)

        if not clean_prompts:
            log.warning("No valid prompts to generate from")
            return []

        try:
            inputs = self.tokenizer(
                clean_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)
        except Exception as e:
            log.error("Tokenizer failed on prompts: %s", e)
            return []

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                do_sample=not Config.DETERMINISTIC,
                temperature=float(temperature),
                top_k=int(top_k) if top_k and top_k > 0 else None,
                top_p=float(top_p),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        results: List[str] = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            gen_tokens = output[input_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results.append(text)
        return results

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into token-bounded chunks.

        Args:
            text (str): Input text to chunk.

        Returns:
            List[str]: List of chunked text segments under token limits.
        """
        if not text or not isinstance(text, str):
            log.warning("Non-string or empty text passed to _chunk_text")
            return []
        text = self._sanitize_text(text)
        if self._count_tokens(text) <= Config.MAX_TOKENS:
            return [text]
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        chunks, current, tokens = [], [], 0
        for p in paras:
            ptoks = self._count_tokens(p)
            if ptoks == 0:
                continue
            if tokens + ptoks > Config.CHUNK_SIZE:
                if current:
                    chunks.append("\n".join(current))
                current, tokens = [p], ptoks
            else:
                current.append(p)
                tokens += ptoks
        if current:
            chunks.append("\n".join(current))
        return chunks

    def summarize(
        self, pdf_path: Path, abstract_text: Optional[str]
    ) -> Optional[Dict[str, str]]:
        """Summarize a single PDF into chosen/rejected outputs.

        Args:
            pdf_path (Path): Path to the PDF file.
            abstract_text (Optional[str]): Optional abstract text fallback.

        Returns:
            Optional[Dict[str, str]]: Dictionary with 'chosen' and 'rejected' summaries; None if extraction fails.
        """
        try:
            text = "\n".join(
                str(page.extract_text() or "") for page in PdfReader(pdf_path).pages
            ).strip()
        except Exception as e:
            log.error("Failed to extract text from %s: %s", pdf_path, e)
            return None

        if not text and Config.USE_ARXIV_ABSTRACTS:
            text = abstract_text or ""
        if not text:
            log.warning("No text available for %s", pdf_path)
            return None

        chunks = self._chunk_text(text)
        if not chunks:
            return None

        chunk_prompts = [self._format_prompt(Config.CHUNK_PROMPT, c) for c in chunks]
        chunk_summaries = self._generate(
            chunk_prompts,
            Config.CHOSEN_TEMPERATURE,
            Config.CHOSEN_TOP_K,
            Config.CHOSEN_TOP_P,
        )

        merged = "\n".join(chunk_summaries)
        chosen_list = self._generate(
            [self._format_prompt(Config.PROMPT_CHOSEN, merged)],
            Config.CHOSEN_TEMPERATURE,
            Config.CHOSEN_TOP_K,
            Config.CHOSEN_TOP_P,
        )
        chosen_summary = chosen_list[0] if chosen_list else ""

        rejected_list = self._generate(
            [self._format_prompt(Config.PROMPT_REJECTED, text)],
            Config.REJECTED_TEMPERATURE,
            Config.REJECTED_TOP_K,
            Config.REJECTED_TOP_P,
        )
        rejected_summary = rejected_list[0] if rejected_list else ""

        return {"chosen": chosen_summary, "rejected": rejected_summary}

    def run(self) -> None:
        """Process all PDFs in the raw folder and write a JSONL reward dataset."""
        pdf_files = list(Config.RAW_FOLDER.glob("*.pdf"))
        if not pdf_files:
            log.warning("No PDF files found.")
            return

        log.info("Processing %d PDF(s)...", len(pdf_files))
        reward_data = []
        for idx, pdf_path in enumerate(pdf_files, start=1):
            log.info("Paper %d/%d: %s", idx, len(pdf_files), pdf_path.name)
            result = self.summarize(pdf_path, None)
            if result:
                reward_data.append(result)
            gc.collect()
            torch.cuda.empty_cache()

        if reward_data:
            with open(Config.REWARD_DATA_FILE, "w", encoding="utf-8") as f:
                for item in reward_data:
                    f.write(json.dumps(item) + "\n")
            log.info(
                "Saved reward dataset to %s", Path(Config.REWARD_DATA_FILE).resolve()
            )


def main() -> None:
    """Entry point for running the paper summarizer as a script."""
    configure_logging()
    try:
        log.info("Starting summarization...")
        PaperSummarizer().run()
        log.info("Done.")
    except Exception as e:
        log.error("Unhandled error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
