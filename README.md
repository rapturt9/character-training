# AI Chat Interface

A Streamlit-based chat application that supports multiple AI providers (OpenAI, Anthropic, OpenRouter) with system prompt management and conversation logging.

## Features

- **Multi-Provider Support**: Choose between OpenAI, Anthropic Claude, or OpenRouter APIs
- **System Prompt Management**: Configure and save system prompts locally
- **Conversation Logging**: All conversations are automatically saved to local JSON files
- **Message Retrieval**: Look up specific messages or entire conversations by ID
- **Persistent Storage**: System prompts and conversations are saved locally

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run streamlit_chat.py
```

2. Configure your settings in the sidebar:
   - Select your AI provider (OpenAI, Anthropic, or OpenRouter)
   - Enter your API key
   - For OpenRouter, specify the model (e.g., `anthropic/claude-3-sonnet`)

3. Set up your system prompt:
   - Enter your system prompt in the text area at the top
   - Save frequently used prompts with a name for later reuse
   - Load saved prompts from the dropdown in the sidebar

4. Start chatting:
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
- `system_prompts.json`: Stores your saved system prompts
- `conversations/`: Directory containing all conversation logs
  - Each conversation is saved as `conversation_{id}.json`

## API Configuration

### Anthropic
- Requires Anthropic API key
- Uses Claude 3 Sonnet model by default

### OpenAI
- Requires OpenAI API key
- Uses GPT-3.5-turbo model by default

### OpenRouter
- Requires OpenRouter API key
- Specify the model you want to use (e.g., `anthropic/claude-3-sonnet`, `openai/gpt-4`)

## Requirements

- Python 3.7+
- streamlit
- anthropic
- openai
- requests

## Notes

- All conversations are stored locally and persist between sessions
- System prompts are saved locally and can be reused across conversations
- Each new conversation gets a unique UUID for identification
- API keys are entered per session and not stored permanently