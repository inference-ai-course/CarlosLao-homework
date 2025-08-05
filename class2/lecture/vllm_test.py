from vllm import LLM

llm = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.7  # Optional: reduce memory usage
)

output = llm.generate("What is the capital of France?")
print(output)