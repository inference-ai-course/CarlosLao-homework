# qlora_fine_tuner.py
"""Fine-tunes a base language model using QLoRA on a synthetic Q&A dataset.

This script loads a base model, applies LoRA adapters, tokenizes the dataset,
and trains the model using Hugging Face's Trainer API.
"""

import gc
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer

from config import Config
from logging_utils import configure_logging

log = logging.getLogger(__name__)


class QLoRAFineTuner:
    """Fine-tunes a base model using QLoRA."""

    def __init__(self):
        """Initialize the fine-tuner with placeholders for model components."""
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer = None
        self.trainer: Optional[SFTTrainer] = None
        self.ds = None

    def train_model(self):
        """Load model, prepare dataset, and run fine-tuning.

        Raises:
            FileNotFoundError: If the synthetic Q&A dataset file does not exist.
        """
        log.info("Starting fine-tuning...")
        if not Path(Config.SYNTH_QA_FILE).is_file():
            raise FileNotFoundError(f"Missing dataset: {Config.SYNTH_QA_FILE}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            Config.BASE_MODEL, device_map="auto"
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=Config.LORA_R,
            target_modules=Config.LORA_TARGETS,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing=Config.GRAD_CHECKPOINT,
            random_state=3407,
            use_rslora=Config.USE_RSLORA,
            loftq_config=None,
        )

        self.ds = load_dataset(
            "json", data_files=str(Config.SYNTH_QA_FILE), split="train"
        )
        self.ds = self.ds.map(
            lambda b: self.tokenizer(b[Config.TEXT_FIELD], truncation=True),
            batched=True,
            num_proc=1,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,  # updated for new TRL API
            train_dataset=self.ds,
            dataset_text_field=Config.TEXT_FIELD,
            args=TrainingArguments(
                output_dir=str(Config.MODEL_DIR),
                per_device_train_batch_size=Config.BATCH,
                gradient_accumulation_steps=Config.GRAD_ACCUM,
                num_train_epochs=Config.EPOCHS,
                learning_rate=Config.LR,
                fp16=Config.FP16,
                bf16=Config.BF16,
                logging_steps=Config.LOG_STEPS,
                save_strategy="epoch",
                dataloader_num_workers=0,
                seed=3407,
            ),
        )

        self.trainer.train()
        self.model.save_pretrained(Config.MODEL_DIR)
        self.tokenizer.save_pretrained(Config.MODEL_DIR)
        log.info(f"Model directory saved at: {Path(Config.MODEL_DIR).resolve()}")
        log.info("Fine-tuning complete.")

    def release_resources(self):
        """Release model, tokenizer, trainer, and dataset from memory."""
        for attr in ("model", "tokenizer", "trainer", "ds"):
            if getattr(self, attr) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def run(self):
        """Run the fine-tuning process and release resources.

        This method wraps the training process in a try/except/finally block to
        ensure that resources are released even if training fails.

        Raises:
            Exception: Propagates any exception raised during training after logging.
        """
        try:
            self.train_model()
        except Exception as e:
            log.error(f"Training failed: {e}", exc_info=True)
            raise
        finally:
            self.release_resources()


def main():
    """Entry point for running the fine-tuner as a script."""
    configure_logging()
    try:
        QLoRAFineTuner().run()
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
