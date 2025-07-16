import os
from openai import OpenAI
from dotenv import load_dotenv


def get_llm_config():
    """Load API key, model name, and base URL from environment."""
    load_dotenv()
    api_key = os.getenv('EVAL_API_KEY')
    if not api_key:
        raise ValueError("Eval API key must be provided via the EVAL_API_KEY environment variable.")

    model = os.getenv('EVAL_LLM')
    if not model:
        raise ValueError("Eval LLM model name must be provided via the EVAL_LLM environment variable.")

    base_url = os.getenv('EVAL_BASE_URL')
    if not base_url:
        raise ValueError("Eval Base URL must be provided via the EVAL_BASE_URL environment variable.")

    return api_key, model, base_url


# Initialize LLM client and config at module load
EVAL_API_KEY, EVAL_MODEL, EVAL_BASE_URL = get_llm_config()
client = OpenAI(api_key=EVAL_API_KEY, base_url=EVAL_BASE_URL)


def compute_machine_score(original_answer: str, predicted_answer: str) -> int:
    """
    Uses the LLM to compare the original and predicted answers for semantic equivalence.
    Returns 1 if semantically match, 0 otherwise.
    """
    system_message = (
        "You are a precise answer evaluator. Your sole task is to determine if two answers match semantically in the context of problems.\n"
        "- Ignore minor differences in formatting, notation, or phrasing that don't alter the mathematical meaning\n"
        "YOUR RESPONSE MUST BE EXACTLY ONE CHARACTER:\n"
        "- Return \"1\" if answers match in mathematical meaning\n"
        "- Return \"0\" if answers do not match\n"
        "Do not include any explanation, reasoning, or additional text.")

    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role":
                    "user",
                "content": (f"Compare these two answers:\n"
                            f"Original Answer: {original_answer}\n"
                            f"Predicted Answer: {predicted_answer}\n"
                            f"Return only 0 or 1."),
            },
        ],
    )

    result = response.choices[0].message.content.strip()
    return int(result)
