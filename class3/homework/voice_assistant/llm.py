"""
llm.py
------
Language Model (LLM) integration module.

Provides the LLMService class for generating text responses
based on a provided prompt using a Hugging Face transformer pipeline.
"""

from transformers import pipeline
from typing import Optional, List, Dict, Any

class LLMService:
    """
    Service for generating AI responses using a specified text generation model.
    """
    def __init__(self, model_id: str, token: str):
        """
        Initialize the LLMService.

        Parameters
        ----------
        model_id : str
            The Hugging Face model identifier.
        token : str
            The authentication token for Hugging Face Hub access.
        """
        self.model_id = model_id
        self.token = token
        self.generator: Optional[any] = None

    def load(self):
        """
        Load the model pipeline into memory.
        """
        self.generator = pipeline(
            "text-generation",
            model=self.model_id,
            token=self.token,
            return_full_text=False
        )

    def generate(self, prompt: str,
                 max_new_tokens: int = 200,
                 temperature: float = 0.7) -> str:
        """
        Generate text from the model given a prompt.

        Parameters
        ----------
        prompt : str
            The input prompt containing context and instructions.
        max_new_tokens : int, optional
            Maximum tokens to generate, by default 200.
        temperature : float, optional
            Sampling temperature; higher values yield more randomness.

        Returns
        -------
        str
            Model-generated continuation text.
        """
        if self.generator is None:
            raise RuntimeError("LLM not loaded")
        outputs: List[Dict[str, Any]] = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature
        )
        text = (outputs[0].get("generated_text") or "").strip()
        return text.lstrip("Assistant:").strip()
