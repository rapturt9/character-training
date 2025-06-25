import os
import uuid
import json
import sqlite3
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import csv
import asyncio
from llm_api import call_llm_api, JudgeDecision

# --- Configuration ---
DATABASE_FILE = "conversations.db"
SYSTEM_PROMPTS_FILE = "system_prompts.json"
FAILURE_MODES_CSV = "failure_modes.csv"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Core Functions ---

def load_system_prompts() -> Dict[str, str]:
    """Load saved system prompts from the JSON file."""
    if os.path.exists(SYSTEM_PROMPTS_FILE):
        with open(SYSTEM_PROMPTS_FILE, 'r') as f:
            return json.load(f)
    logger.warning(f"{SYSTEM_PROMPTS_FILE} not found. No prompts loaded.")
    return {}

def save_conversation_to_db(conversation_id: str, messages: List[Dict], system_prompt: str, model: str, summary: str):
    """Save a complete conversation to the SQLite database."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        provider = model.split('/')[0] if '/' in model else "unknown"
        cursor.execute(
            "INSERT OR REPLACE INTO conversations (id, created_at, system_prompt, provider, model, summary) VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, datetime.now().isoformat(), system_prompt, provider, model, summary)
        )
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        for i, msg in enumerate(messages):
            cursor.execute(
                "INSERT INTO messages (conversation_id, message_index, role, content) VALUES (?, ?, ?, ?)",
                (conversation_id, i, msg['role'], msg['content'])
            )
        conn.commit()
    logger.info(f"Successfully saved conversation {conversation_id} to the database.")

async def summarize_conversation(messages: List[Dict], model: str) -> str:
    """Generate a one-sentence summary of the conversation."""
    summary_prompt = "Summarize the following conversation in ten words or less for use as a title."
    conversation_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
    
    api_messages = [
        {"role": "system", "content": summary_prompt},
        {"role": "user", "content": conversation_text}
    ]
    
    response = await call_llm_api(
        messages=api_messages,
        model=model
    )
    return response.strip()

async def judge_response(
    conversation_history: List[Dict],
    ai_persona: str,
    ai_response: str,
    model: str
) -> Optional[Dict]:
    """
    Judges the AI's response based on persona adherence using a structured output model.
    Returns a dictionary with failure mode and reason if it fails, otherwise returns None.
    """
    judge_prompt = f"""
You are a strict evaluation judge. Your task is to determine if the AI assistant's last response adheres to its persona.

**Conversation History:**
{json.dumps(conversation_history, indent=2)}

**AI Persona:**
{ai_persona}

**AI's Last Response:**
{ai_response}

**Instructions:**
1.  Review the conversation history and the AI's persona.
2.  Assess if the AI's last response is consistent with its persona. A failure occurs if the AI breaks character, becomes too generic, refuses to answer when it should, or provides a canned response.
3.  You must determine if the response is good or not and provide a failure mode and reason if it is not.
"""
    
    system_prompt = "You are a strict evaluation judge. You must return a JSON object that conforms to the JudgeDecision model."
    api_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": judge_prompt}
    ]
    
    decision = await call_llm_api(
        messages=api_messages,
        model=model,
        response_model=JudgeDecision
    )

    if isinstance(decision, JudgeDecision):
        if not decision.good_response:
            return {
                "failure_mode": decision.failure_mode or "Unknown Failure",
                "reason": decision.reason or "No reason provided."
            }
        return None
    else:
        logger.error(f"Judge model failed to return a valid JudgeDecision object. Response: {decision}")
        return None

def save_failure_mode_to_csv(conversation_id: str, message_index: int, ai_response: str, failure_data: Dict):
    """Appends a detected failure mode to the CSV log."""
    file_exists = os.path.isfile(FAILURE_MODES_CSV)
    with open(FAILURE_MODES_CSV, 'a', newline='') as csvfile:
        fieldnames = ['conversation_id', 'message_index', 'ai_response', 'failure_mode', 'reason']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'conversation_id': conversation_id,
            'message_index': message_index,
            'ai_response': ai_response,
            'failure_mode': failure_data['failure_mode'],
            'reason': failure_data['reason']
        })
    logger.info(f"Logged failure mode to {FAILURE_MODES_CSV}")


# --- Conversation Generation Logic ---
async def generate_conversation(
    human_persona: str, 
    ai_persona: str, 
    num_turns: int, 
    human_model: str,
    ai_model: str,
    judge_model: str,
    initial_prompt: str,
    conversation_id: str
) -> Tuple[List[Dict], str]:
    """Simulates a conversation between a human and an AI agent, with judging."""
    messages = []
    
    messages.append({"role": "user", "content": initial_prompt})
    logger.info(f"Human (Turn 0): {initial_prompt}")

    for turn in range(num_turns):
        # AI's turn to respond
        ai_messages = [{"role": "system", "content": ai_persona}] + messages
        ai_response = await call_llm_api(
            messages=ai_messages, 
            model=ai_model
        )
        messages.append({"role": "assistant", "content": ai_response})
        logger.info(f"AI (Turn {turn + 1}): {ai_response[:100].replace('\n', ' ')}...")

        # Judge the AI's response
        failure = await judge_response(
            conversation_history=messages[:-1], # History before the AI's latest response
            ai_persona=ai_persona,
            ai_response=ai_response,
            model=judge_model
        )
        if failure:
            save_failure_mode_to_csv(conversation_id, len(messages) - 1, ai_response, failure)

        # Human's turn to respond
        human_messages = [{"role": "system", "content": human_persona}] + messages
        human_response = await call_llm_api(
            messages=human_messages, 
            model=human_model
        )
        messages.append({"role": "user", "content": human_response})
        logger.info(f"Human (Turn {turn + 1}): {human_response[:100].replace('\n', ' ')}...")

    return messages, ai_persona

# --- Main Execution ---
async def main():
    parser = argparse.ArgumentParser(description="Auto-generate and evaluate conversations between a human persona and an AI assistant.")
    parser.add_argument("--human-persona", required=True, help="The system prompt that defines the human user's personality.")
    parser.add_argument("--ai-persona-name", required=True, help="The name of the saved system prompt to use for the AI assistant.")
    parser.add_argument("--initial-prompt", required=True, help="The first message from the human to start the conversation.")
    parser.add_argument("--num-turns", type=int, default=5, help="Number of back-and-forth turns in the conversation.")
    parser.add_argument("--human-model", default="openai/gpt-4o-mini", help="Model for the human simulator (e.g., 'openai/gpt-4o-mini').")
    parser.add_argument("--ai-model", default="openai/gpt-4o-mini", help="Model for the AI assistant (e.g., 'anthropic/claude-3-haiku-20240307').")
    parser.add_argument("--judge-model", default="groq/llama3-8b-8192", help="Lightweight model for the judge (e.g., 'groq/llama3-8b-8192').")
    
    args = parser.parse_args()

    if not any([os.getenv("OPENAI_API_KEY"), os.getenv("ANTHROPIC_API_KEY"), os.getenv("GROQ_API_KEY"), os.getenv("OPENROUTER_API_KEY")]):
        logger.warning("No common API keys found. The script may fail if the selected models require keys.")

    all_prompts = load_system_prompts()
    if args.ai_persona_name not in all_prompts:
        logger.error(f"Error: AI persona '{args.ai_persona_name}' not found in {SYSTEM_PROMPTS_FILE}.")
        return
    
    ai_persona = all_prompts[args.ai_persona_name]
    conversation_id = str(uuid.uuid4())

    logger.info(f"Starting conversation generation (ID: {conversation_id}) with Human ({args.human_model}) and AI ({args.ai_model})")
    
    generated_messages, system_prompt = await generate_conversation(
        human_persona=args.human_persona,
        ai_persona=ai_persona,
        num_turns=args.num_turns,
        human_model=args.human_model,
        ai_model=args.ai_model,
        judge_model=args.judge_model,
        initial_prompt=args.initial_prompt,
        conversation_id=conversation_id
    )

    logger.info("Generating conversation summary...")
    summary = await summarize_conversation(generated_messages, args.ai_model)
    logger.info(f"Generated Summary: {summary}")

    save_conversation_to_db(
        conversation_id=conversation_id,
        messages=generated_messages,
        system_prompt=system_prompt,
        model=args.ai_model,
        summary=summary
    )

if __name__ == "__main__":
    asyncio.run(main())
