## Conversation Generation and Evaluation

This project includes a powerful script for automatically generating and evaluating conversations between a simulated human and an AI assistant. This is ideal for creating training data and testing an AI's ability to adhere to a specific persona.

### How It Works

The `generate_conversations.py` script orchestrates a three-part simulation, now powered by `litellm` for broad compatibility with any LLM provider:

1.  **Human Simulator**: An LLM-powered agent that plays the role of the human user, guided by a persona you define.
2.  **AI Assistant**: The AI being tested. It uses one of the personas defined in `system_prompts.json`.
3.  **Judge**: A separate, lightweight LLM agent (`groq/llama3-8b-8192` by default) that evaluates every response from the AI Assistant. It checks for persona adherence and logs any failures.

### Structured Failure Logging

When the Judge determines that the AI Assistant has broken character, it responds with a structured JSON object enforced by a Pydantic model. This ensures reliable, machine-readable output. The incident is then recorded in `failure_modes.csv`.

The log includes:

- The conversation ID and message index.
- The AI's response that failed.
- The type of failure (e.g., `Persona Deviation`, `Generic Response`).
- The Judge's reasoning.

### Commands

The script now uses `litellm` syntax for specifying models (`provider/model`).

```bash
python generate_conversations.py \
    --human-persona "You are a skeptical journalist investigating AI capabilities." \
    --ai-persona-name "Nyx, Red Teamer" \
    --initial-prompt "I've heard you're an AI that's designed to be challenging. Prove it." \
    --num-turns 3 \
    --human-model "openai/gpt-4o-mini" \
    --ai-model "anthropic/claude-3-5-haiku-latest" \
    --judge-model "openrouter/mistralai/mistral-small-24b-instruct-2501"
```

### Arguments

- `--human-persona`: (Required) The system prompt defining the human simulator's personality.
- `--ai-persona-name`: (Required) The name of the saved system prompt for the AI assistant.
- `--initial-prompt`: (Required) The first message from the human to start the conversation.
- `--num-turns`: The number of back-and-forth turns.
- `--human-model`: The model for the human simulator. Specify provider and model using `provider/model` format (e.g., `openai/gpt-4o-mini`).
- `--ai-model`: The model for the AI assistant. Specify provider and model using `provider/model` format (e.g., `anthropic/claude-3-haiku-20240307`).
- `--judge-model`: The lightweight model for the evaluation judge. Specify provider and model using `provider/model` format (e.g., `groq/llama3-8b-8192`).

# AI Chat Interface

A Streamlit-based chat application that supports any LLM provider through `litellm`, with system prompt management and conversation logging.

## Features

- **Universal Provider Support**: `litellm` integration allows for using models from any provider (OpenAI, Anthropic, Groq, OpenRouter, etc.).
- **System Prompt Management**: Configure and save system prompts locally.
- **Automated Conversation Generation & Evaluation**: Generate training data and evaluate persona adherence automatically.
- **Conversation Logging**: All conversations are automatically saved to a SQLite database.
- **Structured Failure Mode Analysis**: Judge-identified failures are logged to a CSV for review using a Pydantic model for reliable output.
- **Message Retrieval**: Look up specific messages or entire conversations by ID.
- **Persistent Storage**: System prompts and conversations are saved locally.

## Installation

1.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

2.  Set your API keys as environment variables. `litellm` will automatically detect and use them.

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GROQ_API_KEY="your-groq-key"
export OPENROUTER_API_KEY="your-openrouter-key"
# Add other keys as needed
```

## Usage

1.  Run the Streamlit application:

```bash
streamlit run streamlit_chat.py
```

2.  Configure your settings in the sidebar:

    - Select your desired model from the dropdown. The provider is inferred from the model name (e.g., `openai/gpt-4o-mini`).

3.  Set up your system prompt:

    - Enter your system prompt in the text area at the top
    - Save frequently used prompts with a name for later reuse
    - Load saved prompts from the dropdown in the sidebar

4.  Start chatting:
    - Type your message in the chat input
    - Each message displays with a unique ID for easy reference
    - Conversations are automatically saved

## Message and Conversation Management

### Message IDs

Each message displays an ID in the format: `conversation_id:message_index`

- Copy this ID to reference specific messages later
- Use the message retrieval panel to look up messages

### Conversation Retrieval

In the right panel:

- **Get Message**: Enter conversation ID and message index to retrieve a specific message
- **Get Full Conversation**: Enter conversation ID to retrieve the entire conversation
- **Available Conversations**: Click on any listed conversation to load it

### File Structure

- `conversations.db`: A SQLite database storing all conversations.
- `system_prompts.json`: Stores your saved system prompts.
- `failure_modes.csv`: Logs all identified persona failures from the generation script.
- `conversations_migrated/`: Directory containing all conversation logs after they have been migrated to the database.
