# voice_assistant/llm.py
import logging
from transformers import pipeline

logger = logging.getLogger("voice_assistant")

class LLMService:
    def __init__(self, model_name: str, token: str):
        if not token:
            raise RuntimeError("Missing Hugging Face token. Set HUGGINGFACE_TOKEN.")
        logger.info("Initializing LLM pipeline: %s", model_name)
        self.pipe = pipeline("text-generation", model=model_name, token=token)
        logger.info("LLM pipeline ready.")

    def generate(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
        logger.info("Generating response (prompt_len=%d)", len(prompt))
        outputs = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        full_text = outputs[0]["generated_text"]
        reply = full_text[len(prompt):].strip().split("\n")[0].strip()
        logger.info("Generated reply (chars=%d)", len(reply))
        logger.debug("Reply preview: %s", reply[:200])
        return reply

    def ready(self) -> bool:
        try:
            result = self.pipe("Hello", max_new_tokens=5)
            return bool(result and result[0].get("generated_text"))
        except Exception as e:
            logger.error("LLM readiness failed: %s", e)
            return False
