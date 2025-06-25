import litellm
import logging
from typing import Optional, Type, TypeVar, List, Dict, Any
from pydantic import BaseModel, Field, ValidationError
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for Structured Responses ---

class LLMResponse(BaseModel):
    """A standard response from the LLM."""
    content: str = Field(..., description="The text content of the response from the language model.")

class JudgeDecision(BaseModel):
    """The judge's decision on whether the AI's response was good."""
    failure_mode: Optional[str] = Field(None, description="The type of failure. One of: [\"Persona Deviation\", \"Refusal\", \"Generic Response\", \"Other\"]")
    reason: Optional[str] = Field(None, description="A brief explanation for the failure.")
    good_response: bool = Field(..., description="Whether the AI response adhered to its persona.")

T = TypeVar("T", bound=BaseModel)

def clean_json_string(json_string: str) -> str:
    """
    Cleans a JSON string by removing markdown code fences and correcting common formatting issues.
    """
    # Remove markdown fences (```json ... ``` or ``` ... ```)
    match = re.search(r"```(json)?\s*(.*?)\s*```", json_string, re.DOTALL)
    if match:
        return match.group(2).strip()
    return json_string.strip()

# --- Centralized API Call Function ---

async def call_llm_api(
    messages: List[Dict[str, str]],
    model: str,
    response_model: Optional[Type[T]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> T | str:
    """
    Makes an asynchronous call to an LLM API using litellm.

    Args:
        messages: A list of messages in the conversation.
        model: The model to use for the completion.
        response_model: The Pydantic model to parse the response into.
        temperature: The temperature for sampling.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        If a response_model is provided, returns an instance of that model.
        Otherwise, returns the raw text response as a string.
        Returns an error message string if the API call or parsing fails.
    """
    try:
        # Always get a raw text response first
        raw_response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response_text = raw_response.choices[0].message.content or ""

        if not response_model:
            return response_text

        # If a model is expected, clean and parse the text
        cleaned_text = clean_json_string(response_text)
        
        try:
            return response_model.parse_raw(cleaned_text)
        except (ValidationError, json.JSONDecodeError) as e:
            error_message = f"Failed to parse {response_model.__name__} from {model}: {e}"
            logger.error(error_message)
            # Fallback: return the raw (but cleaned) text for the caller to handle
            return cleaned_text

    except Exception as e:
        error_message = f"Error calling LLM API for model {model}: {e}"
        logger.error(error_message)
        return f"[ERROR: {error_message}]"
