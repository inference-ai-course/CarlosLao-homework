# Example: Using LCEL to reproduce a "Basic Prompting" scenario
import sys
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputKeyToolsParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import Tool

sys.stdout.reconfigure(encoding='utf-8')

def run_lcel_prompt(user_prompt, user_inputs):
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
    run_lcel_prompt(prompt, inputs)

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

    run_lcel_prompt(prompt, inputs)

"""
Summarization
"""
def run_information_extraction():
    prompt = "Extract the age and location from the following text:\n{text}"
    inputs = { "text": "John Doe, a 29-year-old software engineer from San Francisco, recently joined OpenAI as a research scientist." }
    run_lcel_prompt(prompt, inputs)

"""
Transformation
"""
def run_transformation():
    prompt = "Translate the following text to {language}:\n{text}"
    inputs = {
        "language": "Spanish",
        "text": "The weather is nice today."
    }
    
    run_lcel_prompt(prompt, inputs)

"""
Expansion
"""
def run_expansion():
    prompt = "Write a poem about a robot exploring space."
    inputs = {}
    run_lcel_prompt(prompt, inputs)

"""
Role-based Prompting
"""
def run_role_based_prompting():
    prompt = "Explain a complex topic as if it were a kindergarten teacher."
    inputs = {}
    run_lcel_prompt(prompt, inputs)

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
    run_lcel_prompt(prompt, inputs)

"""
Chain-of-Thought Prompting
"""
def run_chain_of_thought_prompting():
    prompt = (
        "If it takes 4 workers 8 hours to build 4 walls, "
        "how long would it take 8 workers to build 8 walls, assuming all workers work at the same rate?"
    )
    inputs = {}
    run_lcel_prompt(prompt, inputs)

"""
System Prompts
"""
def run_system_prompts():
    # 2. Define the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("user", "{user_prompt}")
    ])

    # 3. Define the model
    model = ChatOllama(model = "llama2")  # Using Ollama

    # 4. Passthrough chain for each input key
    inputs = {
        "system_prompt": (
            "You're a witty, humorous assistant who sprinkles clever quips, pop culture references, and playful sarcasm into your helpful answers. "
            "You balance intelligence with charm, aiming to educate, entertain, and occasionally make the user laugh-snort. "
            "Never mean-spirited, always delightful—and you know when to dial the jokes back if things get serious."
        ),
        "user_prompt": "Can you explain the importance of data privacy?"
    }

    # 5. Chain the components together using LCEL
    chain = (
        # LCEL syntax: use the pipe operator | to connect each step
        prompt                          # Transform it into a prompt message
        | model                           # Call the model
        | StrOutputParser()               # Parse the output as a string
    )

    # 6. Execute
    result = chain.invoke(inputs)
    print(prompt.format(**inputs))
    print("Model answer:", result)
    print("\n----------------------------\n")

"""
Utilized prompt
"""
def run_utilized_prompts():
    # 2. Define the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("user", "{user_prompt_1}"),
        ("assistant", "{assistant_prompt}"),
        ("user", "{user_prompt_2}")
    ])

    # 3. Define the model
    model = ChatOllama(model = "llama2")  # Using Ollama

    # 4. Passthrough chain for each input key
    inputs = {
        "system_prompt": "You are a helpful assistant knowledgeable in history.",
        "user_prompt_1": "Who was the first prime minister of Canada?",
        "assistant_prompt": "Sir John A. Macdonald was the first prime minister of Canada.",
        "user_prompt_2": "When did he take office?",
    }

    # 5. Chain the components together using LCEL
    chain = (
        # LCEL syntax: use the pipe operator | to connect each step
        prompt                          # Transform it into a prompt message
        | model                           # Call the model
        | StrOutputParser()               # Parse the output as a string
    )

    # 6. Execute
    result = chain.invoke(inputs)
    print(prompt.format(**inputs))
    print("Model answer:", result)
    print("\n----------------------------\n")


def add_numbers(a, b):
    return a + b

def subtract_numbers(a, b):
    return a - b

def multiply_numbers(a, b):
    return a * b

"""
Creating an AI Agent
"""
def run_create_ai_agent():
    # 2. Define the prompt
    prompt = PromptTemplate.from_template("{question}")

    # 3. Set the tools
    tools = [
        Tool.from_function(
            name="add_numbers",
            description="Add two numbers together",
            func=add_numbers,
        ),
        Tool.from_function(
            name="subtract_numbers",
            description="Subtract one number from another",
            func=subtract_numbers,
        ),
        Tool.from_function(
            name="multiply_numbers",
            description="Multiply two numbers together",
            func=multiply_numbers,
        ),
    ]

    # 4. Define the model
    model = ChatOllama(model = "llama2")  # Using Ollama

    # 5. Set the tool selector
    tool_selector = prompt | model | JsonOutputKeyToolsParser(key_name = "name", tools = tools)

    # 6. Chain the components together using LCEL
    chain = RunnableBranch(
        (lambda x: tool_selector.invoke(x) == "add_numbers", prompt | tools[0]),
        (lambda x: tool_selector.invoke(x) == "subtract_numbers", prompt | tools[1]),
        (lambda x: tool_selector.invoke(x) == "multiply_numbers", prompt | tools[2]),
        prompt | model | StrOutputParser() ,
    )

    # 7. Passthrough chain for each input key
    inputs = {
        "question": "4 times 4 equat to what?"
    }

    # 8. Execute
    result = chain.invoke(inputs)
    print(f"User prompt: {prompt.format(**inputs)}")
    print("Model answer:", result)
    print("\n----------------------------\n")

run_basic_prompting()

run_summarization()

run_information_extraction()

run_transformation()

run_expansion()

run_role_based_prompting()

run_few_shot_prompting()

run_chain_of_thought_prompting()

run_system_prompts()

run_utilized_prompts()

run_create_ai_agent()