import streamlit as st
import json
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional
import sqlite3
import asyncio
from database import (
    init_db,
    save_conversation_to_db,
    load_conversation_from_db,
    get_all_conversations_from_db,
    get_message_by_index_from_db,
    migrate_json_to_sqlite
)
from llm_api import call_llm_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streamlit_chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SYSTEM_PROMPTS_FILE = "system_prompts.json"
CONVERSATIONS_DIR = "conversations" # Keep for migration
DATABASE_FILE = "conversations.db"


# Model options for each provider
ANTHROPIC_MODELS = [
    "anthropic/claude-3-5-haiku-latest",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-opus-4-20250514",
]

OPENAI_MODELS = [
    "openai/gpt-4o-mini",
    "openai/o4-mini"
]

OPENROUTER_MODELS = [
    "openrouter/mistralai/mistral-small-3.2-24b-instruct"
]


def ensure_directories():
    """Ensure required directories exist"""
    if not os.path.exists(CONVERSATIONS_DIR):
        os.makedirs(CONVERSATIONS_DIR)


def load_system_prompts() -> Dict[str, str]:
    """Load saved system prompts from JSON file"""
    if os.path.exists(SYSTEM_PROMPTS_FILE):
        with open(SYSTEM_PROMPTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_system_prompts(prompts: Dict[str, str]):
    """Save system prompts to JSON file"""
    with open(SYSTEM_PROMPTS_FILE, 'w') as f:
        json.dump(prompts, f, indent=2)


def summarize_conversation(messages: List[Dict], model: str) -> str:
    """Summarize a conversation using a specified model via the central API call."""
    # Only include user/assistant messages for summary
    summary_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages if msg["role"] in ("user", "assistant")
    ]
    # Compose a prompt for summarization
    prompt = (
        "Summarize the following conversation in 2-3 sentences for future reference. "
        "Be concise and capture the main topics and tone.\n\n"
        "Conversation:\n"
    )
    for msg in summary_messages:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    try:
        api_messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations for future reference."},
            {"role": "user", "content": prompt}
        ]
        summary = asyncio.run(call_llm_api(
            messages=api_messages,
            model=model
        ))
        return summary.strip()
    except Exception as e:
        logger.error(f"Error summarizing conversation: {e}")
        return "[Summary unavailable due to error]"


def save_conversation(conversation_id: str, messages: List[Dict], system_prompt: str = "", model: str = "", summary: str = ""):
    """Save conversation to the database."""
    provider = model.split('/')[0] if '/' in model else "unknown"
    save_conversation_to_db(conversation_id, messages, system_prompt, provider, model, summary)


def load_conversation(conversation_id: str) -> Optional[Dict]:
    """Load conversation from DB, with a fallback to JSON for backward compatibility."""
    # Try loading from the database first
    conversation = load_conversation_from_db(conversation_id)
    if conversation:
        return conversation
    
    # Fallback to legacy JSON file
    filename = f"{CONVERSATIONS_DIR}/conversation_{conversation_id}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
            
    return None


def get_message_by_index(conversation_id: str, message_index: int) -> Optional[Dict]:
    """Get specific message from conversation by index from the database."""
    return get_message_by_index_from_db(conversation_id, message_index)


def list_available_conversations() -> List[Dict]:
    """List all available conversations from the database."""
    return get_all_conversations_from_db()


def main():
    st.set_page_config(page_title="AI Chat Interface", layout="wide")
    logger.info("--- Starting Streamlit App ---")
    
    # Initialize the database and migrate old files
    init_db()
    logger.info("Database initialized.")
    migrate_json_to_sqlite()
    logger.info("Migration check complete.")
    
    ensure_directories()
    
    # Initialize session state early
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("Initialized 'messages' in session state.")
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
        logger.info("Initialized 'conversation_id' in session state.")
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
        logger.info("Initialized 'system_prompt' in session state.")
    
    # Handle conversation loading from query parameters
    logger.info("Checking for 'load_conv_id' in query parameters.")
    load_conv_id = st.query_params.get("load_conv_id")
    if load_conv_id and load_conv_id != st.session_state.conversation_id:
        logger.info(f"Attempting to load conversation: {load_conv_id}")
        conversation = load_conversation(load_conv_id)
        if conversation:
            st.session_state.messages = conversation["messages"]
            st.session_state.conversation_id = load_conv_id
            logger.info(f"Successfully loaded conversation {load_conv_id}.")
            # Restore system prompt if available
            if "system_prompt" in conversation:
                st.session_state.system_prompt = conversation["system_prompt"]
            # Handle message highlighting
            highlight_msg = st.query_params.get("highlight_msg")
            if highlight_msg:
                try:
                    st.session_state.highlight_message = int(highlight_msg)
                except ValueError:
                    pass
            # Clear query parameters
            st.query_params.clear()
            logger.info("Cleared query parameters.")
    
    st.title("AI Chat Interface")
    logger.info("Rendered page title.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        logger.info("Rendering sidebar.")
        
        # API Provider Selection
        # This is now simplified as the provider is determined from the model
        st.info("Select a model below. The provider (OpenAI, Anthropic, OpenRouter) will be inferred automatically.")
        
        all_models = OPENAI_MODELS + ANTHROPIC_MODELS + OPENROUTER_MODELS
        model = st.selectbox("Select Model", all_models, index=all_models.index("openrouter/mistralai/mistral-small-3.2-24b-instruct"))
        
        # API keys are now handled by the central llm_api.py, which reads from env vars.
        # We can add a note to the user.
        st.caption("Ensure `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `OPENROUTER_API_KEY` are set in your environment variables.")

        st.divider()
        
        # System Prompt Management
        st.header("System Prompts")
        
        # Load saved prompts
        saved_prompts = load_system_prompts()
        
        # Select existing prompt
        prompt_names = list(saved_prompts.keys())
        if prompt_names:
            selected_prompt = st.selectbox("Load Saved Prompt", [""] + prompt_names)
            if selected_prompt and selected_prompt != st.session_state.get("selected_prompt", ""):
                st.session_state.system_prompt = saved_prompts[selected_prompt]
                st.session_state.selected_prompt = selected_prompt
                st.rerun()
        
        # Save current prompt
        if st.session_state.system_prompt:
            prompt_name = st.text_input("Save Prompt As:")
            if st.button("Save Prompt"):
                if prompt_name:
                    saved_prompts[prompt_name] = st.session_state.system_prompt
                    save_system_prompts(saved_prompts)
                    st.success(f"Prompt saved as '{prompt_name}'")
                    st.rerun()
        
        st.divider()

        # Conversation History
        st.header("Conversation History")
        available_conversations = list_available_conversations()
        logger.info(f"Found {len(available_conversations)} available conversations.")
        
        if available_conversations:
            for convo in available_conversations:
                summary = convo.get('summary', 'No summary available')
                timestamp = datetime.fromisoformat(convo['created_at']).strftime('%Y-%m-%d %H:%M')
                
                button_label = f"**{timestamp}**\n_{summary}_"
                if st.button(button_label, key=convo['id']):
                    st.query_params["load_conv_id"] = convo['id']
                    st.rerun()
        else:
            st.write("No past conversations found.")

    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    logger.info("Rendered main columns.")
    
    with col1:
        st.header("Chat")
        
        # System prompt configuration - prominent section
        with st.expander("🎯 System Prompt Configuration", expanded=not bool(st.session_state.system_prompt)):
            st.markdown("**Configure the AI's behavior and personality before starting the conversation.**")
            system_prompt = st.text_area(
                "System Prompt",
                value=st.session_state.system_prompt,
                height=100,
                key="system_prompt",
                placeholder="Enter a system prompt to define the AI's behavior (e.g., 'You are a helpful assistant', etc.)"
            )
        
        # Message ID jump functionality
        message_id_input = st.text_input("Jump to Message ID (format: conversation_id:message_index)")
        if st.button("Load & Jump to Message") and message_id_input:
            try:
                if ":" in message_id_input:
                    conv_id, msg_idx = message_id_input.split(":", 1)
                    msg_idx = int(msg_idx)
                    
                    # Load conversation if different from current
                    if conv_id != st.session_state.conversation_id:
                        conversation = load_conversation(conv_id)
                        if conversation:
                            # Set query parameters to trigger reload with new data
                            st.query_params["load_conv_id"] = conv_id
                            st.query_params["highlight_msg"] = str(msg_idx)
                            st.rerun()
                        else:
                            st.error("Conversation not found")
                    else:
                        # Same conversation, just highlight
                        if 0 <= msg_idx < len(st.session_state.messages):
                            st.session_state.highlight_message = msg_idx
                            st.rerun()
                        else:
                            st.error("Message index out of range")
                else:
                    st.error("Invalid message ID format. Use: conversation_id:message_index")
            except ValueError:
                st.error("Invalid message index. Must be a number.")
        
        # Display conversation ID
        st.info(f"Conversation ID: {st.session_state.conversation_id}")
        
        # Chat messages
        for i, message in enumerate(st.session_state.messages):
            # Check if this message should be highlighted
            is_highlighted = st.session_state.get("highlight_message") == i
            
            with st.chat_message(message["role"]):
                if is_highlighted:
                    st.markdown(f"🔍 **HIGHLIGHTED MESSAGE**")
                    st.markdown(f"<div style='background-color: #b8860b; color: #fff; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107;'>{message['content']}</div>", unsafe_allow_html=True)
                    # Clear highlight after showing
                    if st.session_state.get("highlight_message") == i:
                        st.session_state.highlight_message = None
                else:
                    st.markdown(message["content"])
                st.caption(f"Message ID: {st.session_state.conversation_id}:{i}")
        
        # Chat input
        if prompt := st.chat_input("What would you like to discuss?"):
            # API keys are checked within the call_llm_api function
            
            if not st.session_state.system_prompt.strip():
                st.error("Please set a system prompt before starting the conversation")
                return
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                st.caption(f"Message ID: {st.session_state.conversation_id}:{len(st.session_state.messages)-1}")
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Prepare messages for the API call, including the system prompt
                        api_messages = [{"role": "system", "content": st.session_state.system_prompt}] + st.session_state.messages
                        
                        response = asyncio.run(call_llm_api(
                            messages=api_messages,
                            model=model
                        ))
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.caption(f"Message ID: {st.session_state.conversation_id}:{len(st.session_state.messages)-1}")
                        
                        # Save conversation (now with provider and model, summary is not set here)
                        save_conversation(
                            st.session_state.conversation_id,
                            st.session_state.messages,
                            st.session_state.system_prompt,
                            model,
                            summary=""
                        )
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # New conversation button
        if st.button("Start New Conversation"):
            # Summarize previous conversation if it exists and has messages
            if st.session_state.messages:
                # We can use any capable model for summarization. Let's default to a fast one.
                summary_model = "anthropic/claude-3-5-haiku-latest"
                summary = summarize_conversation(
                    st.session_state.messages, summary_model
                )
                
                # Determine provider for logging
                provider = model.split('/')[0] if '/' in model else "Unknown"

                # Save summary to previous conversation
                save_conversation(
                    st.session_state.conversation_id,
                    st.session_state.messages,
                    st.session_state.system_prompt,
                    model,
                    summary=summary
                )
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()
    
    with col2:
        st.header("Tools")
        logger.info("Rendering tools column.")
        
        # Message and Conversation Retrieval Panel
        with st.expander("🔍 Message & Conversation Retrieval", expanded=False):
            st.markdown("**Retrieve specific messages or full conversations by ID.**")
            
            # Get specific message
            st.subheader("Get Message by ID")
            msg_id_input = st.text_input("Enter Message ID (e.g., `conversation_id:message_index`)")
            if st.button("Get Message"):
                try:
                    conv_id, msg_idx = msg_id_input.split(':')
                    message = get_message_by_index(conv_id, int(msg_idx))
                    if message:
                        st.json(message)
                    else:
                        st.error("Message not found.")
                except (ValueError, IndexError):
                    st.error("Invalid format. Use `conversation_id:message_index`.")

            # Get full conversation
            st.subheader("Get Full Conversation")
            conv_id_input = st.text_input("Enter Conversation ID")
            if st.button("Get Full Conversation"):
                conversation = load_conversation(conv_id_input)
                if conversation:
                    st.json(conversation)
                else:
                    st.error("Conversation not found.")

    logger.info("--- Finished Rendering UI ---")

if __name__ == "__main__":
    main()