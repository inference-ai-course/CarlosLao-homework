# evaluation_runner.py
"""Evaluates generated summaries using ROUGE, BERTScore, and a reward model.

This script computes text metrics for generated summaries against references,
scores them with a trained reward model, and saves a JSON report.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from config import Config
from evaluate import load
from logging_utils import configure_logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger(__name__)


class EvaluationRunner:
    """Runs metric and reward-model evaluation on generated summaries."""

    def __init__(self) -> None:
        """Initialize metrics, load reward model artifacts, and select device."""
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")

        model_dir = Path(Config.REWARD_OUTPUT_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        device = (
            Config.DEVICE
            if getattr(Config, "DEVICE", "auto") != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(device)
        self.device = device
        self.model.eval()

    def compute_metrics(
        self, generated: List[str], references: List[str]
    ) -> Tuple[dict, dict, dict]:
        """Compute ROUGE and BERTScore metrics between summaries and references.

        Args:
            generated (List[str]): Generated summaries.
            references (List[str]): Reference summaries.

        Returns:
            Tuple[dict, dict, dict]: Aggregated ROUGE, raw BERTScore dict, and average BERTScore metrics.

        Raises:
            ValueError: If the input lists have mismatched lengths.
        """
        if len(generated) != len(references):
            raise ValueError(
                f"Mismatched input lengths: {len(generated)} generated vs {len(references)} references"
            )

        results_rouge = self.rouge.compute(predictions=generated, references=references)
        results_bertscore = self.bertscore.compute(
            predictions=generated, references=references, lang="en"
        )

        bert_avg = {
            k: round(sum(v) / len(v), 4)
            for k, v in results_bertscore.items()
            if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)
        }

        log.info("ROUGE: %s", results_rouge)
        log.info("BERTScore avg: %s", bert_avg)
        log.debug(
            "BERTScore sample values: %s",
            {k: v[:3] for k, v in results_bertscore.items() if isinstance(v, list)},
        )

        return results_rouge, results_bertscore, bert_avg

    def score_with_reward_model(self, summaries: List[str]) -> List[float]:
        """Score summaries using the trained reward model.

        Args:
            summaries (List[str]): Candidate summaries to score.

        Returns:
            List[float]: Reward scores (logits) for each summary.
        """
        encodings = self.tokenizer(
            summaries,
            truncation=True,
            padding=True,
            max_length=Config.REWARD_MAX_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
            scores = outputs.logits.squeeze(-1).tolist()
        log.info("Reward model scores: %s", scores)
        return scores

    def run(self, generated: List[str], references: List[str]) -> dict:
        """Compute metrics, reward scores, and identify top summaries.

        Args:
            generated (List[str]): Generated summaries.
            references (List[str]): Reference summaries.

        Returns:
            dict: Evaluation results including metrics and reward scores.
        """
        log.info("Computing ROUGE and BERTScore...")
        results_rouge, bert_raw, bert_avg = self.compute_metrics(generated, references)
        rouge_per_example = self.rouge.compute(
            predictions=generated, references=references, use_aggregator=False
        )

        log.info("Scoring with reward model...")
        reward_scores = self.score_with_reward_model(generated)

        top_rouge_idx = max(
            range(len(generated)), key=lambda i: rouge_per_example["rougeL"][i]
        )
        top_reward_idx = max(range(len(generated)), key=lambda i: reward_scores[i])

        log.info(
            "Top by ROUGE: idx=%d | summary=%s", top_rouge_idx, generated[top_rouge_idx]
        )
        log.info(
            "Top by Reward Model: idx=%d | summary=%s",
            top_reward_idx,
            generated[top_reward_idx],
        )

        results = {
            "rouge": results_rouge,
            "bertscore_raw": bert_raw,
            "bertscore_avg": bert_avg,
            "reward_scores": reward_scores,
            "top_by_rouge": {"idx": top_rouge_idx, "summary": generated[top_rouge_idx]},
            "top_by_reward": {
                "idx": top_reward_idx,
                "summary": generated[top_reward_idx],
            },
        }
        return results


def main() -> None:
    """Entry point for running the evaluation as a script."""
    configure_logging()
    try:
        generated_summaries = [
            "This is a generated summary of the paper.",
            "Another candidate summary with different wording.",
        ]
        reference_summaries = [
            "This is the gold reference summary of the paper.",
            "Another gold reference summary for comparison.",
        ]
        runner = EvaluationRunner()
        results = runner.run(generated_summaries, reference_summaries)

        output_path = Path(
            getattr(Config, "EVAL_RESULTS_PATH", "evaluation_results.json")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        log.info("Saved evaluation results to %s", output_path.resolve())

    except Exception as e:
        log.error("Unhandled error in EvaluationRunner: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
