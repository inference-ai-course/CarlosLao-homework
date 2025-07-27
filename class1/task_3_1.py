# Example: Using LCEL to reproduce a "Basic Prompting" scenario
import sys
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama 

sys.stdout.reconfigure(encoding='utf-8')

def run_lcel(user_prompt: str, user_inputs: dict):
    # 2. Define the prompt
    prompt = PromptTemplate.from_template(user_prompt)

    # 3. Define the model
    model = ChatOllama(model = "llama2")  # Using Ollama

    # 4. Passthrough chain for each input key
    inputs = {key: RunnablePassthrough() for key in user_inputs}

    # 5. Chain the components together using LCEL
    chain = (
        # LCEL syntax: use the pipe operator | to connect each step
        inputs                       # Accept user input
        | prompt                          # Transform it into a prompt message
        | model                           # Call the model
        | StrOutputParser()               # Parse the output as a string
    )

    # 6. Execute
    result = chain.invoke(user_inputs)
    print(f"User prompt: {user_prompt.format(**user_inputs)}")
    print("Model answer:", result)
    print("\n----------------------------\n")

"""
Basic Prompting
"""
def run_basic_prompting():
    prompt = "What is the capital of {country}?"
    inputs = {"country": "Germany"}
    run_lcel(prompt, inputs)

"""
Summarization
"""
def run_summarization():
    prompt = "Summarize the following text:\n{text}"
    inputs = {
        "text": (
            "In the face of adversity, resilience isn't just about bouncing back—it's about transforming. "
            "Like roots pushing through stone, growth often happens in resistance. Each challenge becomes a teacher; "
            "each setback, a foundation for reinvention. Whether it's in personal loss, career disruption, or even global turmoil, "
            "those who rise aren't untouched by hardship—they are shaped by it. They emerge with deeper empathy, sharper purpose, "
            "and wider perspective. True resilience is quiet, steady, and often invisible until the bloom appears. And when it does, "
            "it speaks of strength not just endured—but chosen."
        )
    }

    run_lcel(prompt, inputs)

"""
Summarization
"""
def run_information_extraction():
    prompt = "Extract the age and location from the following text:\n{text}"
    inputs = { "text": "John Doe, a 29-year-old software engineer from San Francisco, recently joined OpenAI as a research scientist." }
    run_lcel(prompt, inputs)

"""
Transformation
"""
def run_transformation():
    prompt = "Translate the following text to {language}:\n{text}"
    inputs = {
        "language": "Spanish",
        "text": "The weather is nice today."
    }
    
    run_lcel(prompt, inputs)

"""
Expansion
"""
def run_expansion():
    prompt = "Write a poem about a robot exploring space."
    inputs = {}
    run_lcel(prompt, inputs)

"""
Role-based Prompting
"""
def run_role_based_prompting():
    prompt = "Explain a complex topic as if it were a kindergarten teacher."
    inputs = {}
    run_lcel(prompt, inputs)

"""
Few-shot Prompting
"""
def run_few_shot_prompting():
    prompt = """
        Translate the following English phrases to Chinese:

        English: Good morning
        Chinese: 早安

        English: Good afternoon
        Chinese: 午安

        English: Good night
        Chinese:
        """
    inputs = {}
    run_lcel(prompt, inputs)

run_basic_prompting()

run_summarization()

run_information_extraction()

run_transformation()

run_expansion()

run_role_based_prompting()

run_few_shot_prompting()