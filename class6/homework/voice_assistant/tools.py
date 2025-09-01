# tools.py
import sympy as sp

def calculate(expression: str) -> str:
    try:
        result = sp.sympify(expression).evalf()
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

def search_arxiv(query: str) -> str:
    # Simulated arXiv search result
    return f"Simulated arXiv result for query: '{query}'"
