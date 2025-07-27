# Example: Using LCEL to reproduce a "Basic Prompting" scenario
import gradio
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama 

def run_lcel(user_inputs):
    # 2. Define the prompt
    prompt = PromptTemplate.from_template(
        "What is the capital of {topic}?"
    )

    # 3. Define the model
    model = ChatOllama(model = "llama2")  # Using Ollama 

    # 4. Chain the components together using LCEL
    chain = (
        # LCEL syntax: use the pipe operator | to connect each step
        prompt                          # Transform it into a prompt message
        | model                           # Call the model
        | StrOutputParser()               # Parse the output as a string
    )

    # 5. Execute
    return chain.invoke({"topic": user_inputs})

demo = gradio.Interface(
    fn = run_lcel,
    inputs = gradio.Textbox(label="Country"),
    outputs = gradio.Textbox(label="Result"),
    title = "Advance Work: Integrate the Ollama and Langchain tasks into Gradio Web UI Demo"
)

if __name__ == "__main__":
    demo.launch()