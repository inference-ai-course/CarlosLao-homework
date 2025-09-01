"""
llm.py
------
Language Model (LLM) integration module.

Provides the LLMService class for generating text responses
based on a provided prompt using a Hugging Face transformer pipeline.
Includes tool-call detection and invocation.
"""

from transformers import pipeline
from typing import Optional, List, Dict, Any
import json

from .tools import calculate, search_arxiv
from .prompt import DEFAULT_SYSTEM_PROMPT

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

    def generate(self, user_input: str,
                 max_new_tokens: int = 200,
                 temperature: float = 0.7) -> str:
        """
        Generate text from the model given a prompt.

        Parameters
        ----------
        user_input : str
            The user's input message.
        max_new_tokens : int, optional
            Maximum tokens to generate, by default 200.
        temperature : float, optional
            Sampling temperature; higher values yield more randomness.

        Returns
        -------
        str
            Final assistant reply (tool result or normal text).
        """
        if self.generator is None:
            raise RuntimeError("LLM not loaded")

        full_prompt = f"{DEFAULT_SYSTEM_PROMPT}\nUser: {user_input}\nAssistant:"
        outputs: List[Dict[str, Any]] = self.generator(
            full_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature
        )
        raw_text = (outputs[0].get("generated_text") or "").strip()
        response = raw_text.lstrip("Assistant:").strip()

        return self._handle_response(response)

    def _handle_response(self, response: str) -> str:
        """
        Detect and invoke tool functions if the response is a JSON object.
        Otherwise, return the raw string response.

        Parameters
        ----------
        response : str
            Raw model output (could be JSON or plain text).

        Returns
        -------
        str
            Final assistant reply.
        """
        response = response.strip()

        # Try parsing only if it looks like JSON
        if response.startswith("{") and response.endswith("}"):
            try:
                parsed = json.loads(response)
                func = parsed.get("function")
                args = parsed.get("arguments", {})

                if func == "calculate":
                    return calculate(args.get("expression", ""))
                elif func == "search_arxiv":
                    return search_arxiv(args.get("query", ""))
                else:
                    return response  # Unknown function, return raw
            except json.JSONDecodeError:
                pass  # Fall through to return raw response

        # If not JSON or parsing failed, return as-is
        return response
