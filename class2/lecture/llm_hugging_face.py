import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Hugging Face token from .env file
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables.")

# Model and tokenizer identifiers
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

# Load model with automatic device mapping and precision
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    device_map="auto",  # Automatically selects GPU if available
    torch_dtype="auto",  # Uses float16 on GPU, float32 on CPU
)

# Define prompt
prompt = "The Eiffel Tower is located in"

# Tokenize input and move to model device
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {key: value.to(model.device) for key, value in inputs.items()}

# Generate output
outputs = model.generate(
    **inputs,
    max_new_tokens=10,
    do_sample=False  # Deterministic output; set to True for sampling
)

# Decode and display result
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
