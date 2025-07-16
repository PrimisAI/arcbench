import os
from openai import OpenAI
from dotenv import load_dotenv


def get_llm_config():
    """Load API key, model name, and base URL from environment."""
    load_dotenv()
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key must be provided via the API_KEY environment variable.")

    model = os.getenv('LLM')
    if not model:
        raise ValueError("LLM model name must be provided via the LLM environment variable.")

    base_url = os.getenv('BASE_URL')
    if not base_url:
        raise ValueError("Base URL must be provided via the BASE_URL environment variable.")

    return api_key, model, base_url


API_KEY, MODEL, BASE_URL = get_llm_config()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def ask_baseline_llm(query: str) -> str:
    """
    Sends a prompt to the baseline LLM and returns its answer as a string.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role":
                    "system",
                "content": (
                    "You are a helpful and knowledgeable assistant. "
                    "Your task is to carefully analyze the user's question and provide a clear, concise, and accurate response."
                ),
            },
            {
                "role": "user",
                "content": query
            },
        ],
    )
    return response.choices[0].message.content.strip()
