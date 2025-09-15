# qa_generator.py
"""Generates a synthetic Q&A dataset from grouped paper summaries.

This script reads grouped paper summaries with associated Q&A pairs from a JSON
file and converts them into a prompt-completion format suitable for fine-tuning.
"""

import json
import logging
from pathlib import Path

from config import Config
from logging_utils import configure_logging

log = logging.getLogger(__name__)


class QAGenerator:
    """Generates Q&A entries in prompt-completion format."""

    def run(self):
        """Read grouped Q&A data and write them to a JSONL file.

        Raises:
            FileNotFoundError: If the input JSON file does not exist.
        """
        log.info("Starting Q&A generation...")
        input_path = Path(Config.QA_INPUT_FILE)
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing input file: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            groups = json.load(f)
        if isinstance(groups, dict):
            groups = [groups]

        entries = []
        for group in groups:
            for paper in group.get("papers", []):
                for qa in paper.get("qa_pairs", []):
                    q, a = qa.get("question"), qa.get("answer")
                    if q and a:
                        entries.append(
                            {
                                "text": f"<|system|>{Config.EVAL_PROMPT}<|user|>{q}<|assistant|>{a}"
                            }
                        )

        output_path = Path(Config.SYNTH_QA_FILE)
        with open(output_path, "w", encoding="utf-8") as out:
            for entry in entries:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        log.info(f"Generated {len(entries)} Q&A entries.")
        log.info(f"Output file saved at: {output_path.resolve()}")
        log.info("Q&A generation complete.")


def main():
    """Entry point for running the Q&A generator as a script."""
    configure_logging()
    try:
        QAGenerator().run()
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
