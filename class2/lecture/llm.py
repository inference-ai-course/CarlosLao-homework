import requests
import json

# Prompt to test
prompt = "Explain why the sky is blue in simple terms."


# Get response from Local LLM
def get_local_llm_response(model, prompt):
    url = "http://localhost:11434/api/chat"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(url, json=payload, stream=True)

    # Collect streamed chunks
    full_content = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "message" in data and "content" in data["message"]:
                    full_content += data["message"]["content"]
            except Exception as e:
                print("Error decoding line:", e)

    return full_content if full_content else "No response received."


# Run comparison
local_llm_reply = get_local_llm_response("llama3", prompt)
print("\nLocal LLM(llama3) Response:\n", local_llm_reply)
chatgpt_reply = get_local_llm_response("llama2", prompt)
print("\nLocal LLM(llama2) Response:\n", chatgpt_reply)
