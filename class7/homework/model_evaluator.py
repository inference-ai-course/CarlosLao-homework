# model_evaluator.py
"""Evaluates a fine-tuned model against a base model using Q&A pairs.

This script loads evaluation Q&A pairs, generates answers from both the base
and fine-tuned models, compares them to reference answers using string
similarity, and saves the results to a JSON file.
"""

import gc
import json
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Union

import torch
from config import Config
from unsloth import FastLanguageModel
from logging_utils import configure_logging

log = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates base and fine-tuned models on a Q&A dataset."""

    def __init__(self):
        """Initialize the evaluator with empty data structures."""
        self.qa_pairs: List[Dict[str, str]] = []
        self.base_answers: List[str] = []
        self.ft_answers: List[str] = []
        self.results: List[Dict[str, Union[str, float]]] = []

    def load_qa_pairs(self) -> None:
        """Load evaluation Q&A pairs from a JSONL file.

        Raises:
            FileNotFoundError: If the evaluation file does not exist.
            ValueError: If no valid Q&A pairs are found.
        """
        path = Path(Config.EVAL_QA_FILE)
        if not path.is_file():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        self.qa_pairs = [
            obj
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for obj in [json.loads(line)]
            if "question" in obj and "answer" in obj
        ]
        if not self.qa_pairs:
            raise ValueError("No valid Q&A pairs found.")
        log.info(f"Loaded {len(self.qa_pairs)} Q&A pairs.")

    def generate_answers_for_model(
        self, model_path: Path | str, target_attr: str
    ) -> None:
        """Generate answers for all Q&A pairs using a given model.

        Args:
            model_path (Path | str): Path to the model directory or model name.
            target_attr (str): Attribute name to store generated answers in.
        """
        log.info(f"Generating answers with model: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            str(model_path), device_map="auto"
        )
        answers = []
        for qa in self.qa_pairs:
            prompt = (
                f"<|system|>{Config.EVAL_PROMPT}<|user|>{qa['question']}<|assistant|>"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, max_new_tokens=Config.EVAL_MAX_TOKENS, do_sample=False
            )
            ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "<|assistant|>" in ans:
                ans = ans.split("<|assistant|>", 1)[-1].strip()
            answers.append(ans)
        setattr(self, target_attr, answers)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def string_similarity(a: str, b: str) -> float:
        """Compute the similarity ratio between two strings.

        Args:
            a (str): First string.
            b (str): Second string.

        Returns:
            float: Similarity ratio between 0 and 1.
        """
        return SequenceMatcher(None, a, b).ratio()

    def build_results(self) -> None:
        """Build the evaluation results comparing base and fine-tuned models."""
        self.results = []
        base_total, ft_total = 0.0, 0.0
        for qa, base_ans, ft_ans in zip(
            self.qa_pairs, self.base_answers, self.ft_answers
        ):
            ref = qa["answer"]
            base_sim = self.string_similarity(base_ans, ref)
            ft_sim = self.string_similarity(ft_ans, ref)
            base_total += base_sim
            ft_total += ft_sim
            conclusion = (
                "Base model performed better"
                if base_sim > ft_sim
                else (
                    "Fine-tuned model performed better"
                    if ft_sim > base_sim
                    else "Both models performed equally"
                )
            )
            self.results.append(
                {
                    "question": qa["question"],
                    "reference_answer": ref,
                    "base_answer": base_ans,
                    "finetuned_answer": ft_ans,
                    "base_similarity": base_sim,
                    "finetuned_similarity": ft_sim,
                    "conclusion": conclusion,
                }
            )
        base_avg = base_total / len(self.qa_pairs)
        ft_avg = ft_total / len(self.qa_pairs)
        overall_conclusion = (
            "Base model performed better overall"
            if base_avg > ft_avg
            else (
                "Fine-tuned model performed better overall"
                if ft_avg > base_avg
                else "Both models performed equally overall"
            )
        )
        self.results.append(
            {
                "overall_base_avg_similarity": base_avg,
                "overall_finetuned_avg_similarity": ft_avg,
                "overall_conclusion": overall_conclusion,
            }
        )

    def save_results(self, print_results: bool = False) -> None:
        """Save evaluation results to a JSON file.

        Args:
            print_results (bool): If True, log the full results after saving.
        """
        output_path = Path(Config.EVAL_OUTPUT_FILE)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        log.info(f"Output file saved at: {output_path.resolve()}")
        if print_results:
            log.info("Full evaluation results:")
            log.info(json.dumps(self.results, indent=2, ensure_ascii=False))

    def run(self, print_results: bool = False) -> None:
        """Run the evaluation process.

        Args:
            print_results (bool): If True, log the full results after saving.
        """
        log.info("Starting model evaluation...")
        self.load_qa_pairs()
        self.generate_answers_for_model(Config.BASE_MODEL, "base_answers")
        self.generate_answers_for_model(Config.MODEL_DIR, "ft_answers")
        self.build_results()
        self.save_results(print_results=print_results)
        log.info("Model evaluation complete.")


def main():
    """Entry point for running the model evaluator as a script."""
    configure_logging()
    try:
        ModelEvaluator().run(print_results=True)
    except Exception as e:
        log.error(f"Evaluation failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
