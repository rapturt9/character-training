import streamlit as st
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import anthropic
import openai
import requests

# Configuration
SYSTEM_PROMPTS_FILE = "system_prompts.json"
CONVERSATIONS_DIR = "conversations"

# Model options for each provider
ANTHROPIC_MODELS = [
    "claude-3-5-haiku-latest",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
]

OPENAI_MODELS = [
    "gpt-4o-mini",
    "o4-mini"
]

OPENROUTER_MODELS = [
    "allenai/olmo-2-0325-32b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct"
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

def save_conversation(conversation_id: str, messages: List[Dict]):
    """Save conversation to local file"""
    filename = f"{CONVERSATIONS_DIR}/conversation_{conversation_id}.json"
    conversation_data = {
        "id": conversation_id,
        "created_at": datetime.now().isoformat(),
        "messages": messages
    }
    with open(filename, 'w') as f:
        json.dump(conversation_data, f, indent=2)

def load_conversation(conversation_id: str) -> Optional[Dict]:
    """Load conversation from local file"""
    filename = f"{CONVERSATIONS_DIR}/conversation_{conversation_id}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def get_message_by_index(conversation_id: str, message_index: int) -> Optional[Dict]:
    """Get specific message from conversation by index"""
    conversation = load_conversation(conversation_id)
    if conversation and 0 <= message_index < len(conversation["messages"]):
        return conversation["messages"][message_index]
    return None

def call_anthropic_api(messages: List[Dict], system_prompt: str, api_key: str, model: str) -> str:
    """Call Anthropic Claude API"""
    client = anthropic.Anthropic(api_key=api_key)
    
    formatted_messages = []
    for msg in messages:
        if msg["role"] != "system":
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
    
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=system_prompt,
        messages=formatted_messages
    )
    return response.content[0].text

def call_openai_api(messages: List[Dict], system_prompt: str, api_key: str, model: str) -> str:
    """Call OpenAI API"""
    client = openai.OpenAI(api_key=api_key)
    
    formatted_messages = [{"role": "system", "content": system_prompt}]
    formatted_messages.extend(messages)
    
    response = client.chat.completions.create(
        model=model,
        messages=formatted_messages
    )
    return response.choices[0].message.content

def call_openrouter_api(messages: List[Dict], system_prompt: str, api_key: str, model: str) -> str:
    """Call OpenRouter API"""
    formatted_messages = [{"role": "system", "content": system_prompt}]
    formatted_messages.extend(messages)
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": formatted_messages
        }
    )
    return response.json()["choices"][0]["message"]["content"]

def main():
    st.set_page_config(page_title="AI Chat Interface", layout="wide")
    
    ensure_directories()
    
    st.title("AI Chat Interface")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Provider Selection
        provider = st.selectbox("Select API Provider", ["Anthropic", "OpenAI", "OpenRouter"])
        
        # Get API key from environment or user input
        if provider == "Anthropic":
            env_key = os.getenv("ANTHROPIC_API_KEY")
            api_key = st.text_input("API Key", value=env_key if env_key else "", type="password")
            model = st.selectbox("Model", ANTHROPIC_MODELS)
        elif provider == "OpenAI":
            env_key = os.getenv("OPENAI_API_KEY")
            api_key = st.text_input("API Key", value=env_key if env_key else "", type="password")
            model = st.selectbox("Model", OPENAI_MODELS)
        elif provider == "OpenRouter":
            env_key = os.getenv("OPENROUTER_API_KEY")
            api_key = st.text_input("API Key", value=env_key if env_key else "", type="password")
            model = st.selectbox("Model", OPENROUTER_MODELS)
        
        st.divider()
        
        # System Prompt Management
        st.header("System Prompts")
        
        # Load saved prompts
        saved_prompts = load_system_prompts()
        
        # Select existing prompt
        prompt_names = list(saved_prompts.keys())
        if prompt_names:
            selected_prompt = st.selectbox("Load Saved Prompt", [""] + prompt_names)
            if selected_prompt:
                st.session_state.system_prompt = saved_prompts[selected_prompt]
        
        # Save current prompt
        if "system_prompt" in st.session_state and st.session_state.system_prompt:
            prompt_name = st.text_input("Save Prompt As:")
            if st.button("Save Prompt"):
                if prompt_name:
                    saved_prompts[prompt_name] = st.session_state.system_prompt
                    save_system_prompts(saved_prompts)
                    st.success(f"Prompt saved as '{prompt_name}'")
                    st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chat")
        
        # System prompt input
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.get("system_prompt", ""),
            height=100,
            key="system_prompt"
        )
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        
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
                            st.session_state.messages = conversation["messages"]
                            st.session_state.conversation_id = conv_id
                            st.session_state.highlight_message = msg_idx
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
                    st.markdown(f"<div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107;'>{message['content']}</div>", unsafe_allow_html=True)
                    # Clear highlight after showing
                    if st.session_state.get("highlight_message") == i:
                        st.session_state.highlight_message = None
                else:
                    st.markdown(message["content"])
                st.caption(f"Message ID: {st.session_state.conversation_id}:{i}")
        
        # Chat input
        if prompt := st.chat_input("What would you like to discuss?"):
            if not api_key:
                st.error("Please enter your API key in the sidebar")
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
                        if provider == "Anthropic":
                            response = call_anthropic_api(st.session_state.messages, system_prompt, api_key, model)
                        elif provider == "OpenAI":
                            response = call_openai_api(st.session_state.messages, system_prompt, api_key, model)
                        elif provider == "OpenRouter":
                            response = call_openrouter_api(st.session_state.messages, system_prompt, api_key, model)
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.caption(f"Message ID: {st.session_state.conversation_id}:{len(st.session_state.messages)-1}")
                        
                        # Save conversation
                        save_conversation(st.session_state.conversation_id, st.session_state.messages)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # New conversation button
        if st.button("Start New Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()
    
    with col2:
        st.header("Message Retrieval")
        
        # Chat ID and message lookup
        lookup_conversation_id = st.text_input("Conversation ID")
        message_index = st.number_input("Message Index", min_value=0, value=0)
        
        if st.button("Get Message"):
            if lookup_conversation_id:
                message = get_message_by_index(lookup_conversation_id, message_index)
                if message:
                    st.success("Message found:")
                    st.json(message)
                else:
                    st.error("Message not found")
        
        if st.button("Get Full Conversation"):
            if lookup_conversation_id:
                conversation = load_conversation(lookup_conversation_id)
                if conversation:
                    st.success("Conversation found:")
                    st.json(conversation)
                else:
                    st.error("Conversation not found")
        
        # List all conversations
        st.subheader("Available Conversations")
        if os.path.exists(CONVERSATIONS_DIR):
            conversation_files = [f for f in os.listdir(CONVERSATIONS_DIR) if f.endswith('.json')]
            for file in conversation_files:
                conv_id = file.replace('conversation_', '').replace('.json', '')
                if st.button(f"Load {conv_id[:8]}..."):
                    conversation = load_conversation(conv_id)
                    if conversation:
                        st.session_state.messages = conversation["messages"]
                        st.session_state.conversation_id = conv_id
                        st.rerun()

if __name__ == "__main__":
    main()