import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

DATABASE_FILE = "conversations.db"
CONVERSATIONS_DIR = "conversations"

def init_db():
    """Initialize the database and create tables if they don't exist."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            system_prompt TEXT,
            provider TEXT,
            model TEXT,
            summary TEXT
        )
        """)
        
        # Create messages table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            message_index INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        """)
        
        conn.commit()

def save_conversation_to_db(conversation_id: str, messages: List[Dict], system_prompt: str, provider: str, model: str, summary: str):
    """Save or update a conversation in the database, preserving the original creation time."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        
        # Use COALESCE to keep the original created_at timestamp if it exists
        cursor.execute("""
        INSERT OR REPLACE INTO conversations (id, created_at, system_prompt, provider, model, summary)
        VALUES (?, COALESCE((SELECT created_at FROM conversations WHERE id = ?), ?), ?, ?, ?, ?)
        """, (conversation_id, conversation_id, datetime.now().isoformat(), system_prompt, provider, model, summary))
        
        # Delete existing messages for this conversation to avoid duplicates
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        
        # Insert new messages
        for i, msg in enumerate(messages):
            cursor.execute("""
            INSERT INTO messages (conversation_id, message_index, role, content)
            VALUES (?, ?, ?, ?)
            """, (conversation_id, i, msg['role'], msg['content']))
            
        conn.commit()

def load_conversation_from_db(conversation_id: str) -> Optional[Dict]:
    """Load a full conversation from the database."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Fetch conversation metadata
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        convo_row = cursor.fetchone()
        
        if not convo_row:
            return None
            
        conversation = dict(convo_row)
        
        # Fetch messages
        cursor.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY message_index", (conversation_id,))
        messages = [dict(row) for row in cursor.fetchall()]
        
        conversation['messages'] = messages
        return conversation

def get_all_conversations_from_db() -> List[Dict]:
    """Retrieve all conversations with their summaries."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, created_at, summary FROM conversations ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

def get_message_by_index_from_db(conversation_id: str, message_index: int) -> Optional[Dict]:
    """Get a specific message by its index from the database."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
        SELECT role, content FROM messages 
        WHERE conversation_id = ? AND message_index = ?
        """, (conversation_id, message_index))
        row = cursor.fetchone()
        return dict(row) if row else None

def migrate_json_to_sqlite():
    """Migrate conversations from JSON files to the SQLite database and rename the folder to prevent re-migration."""
    migrated_dir_name = f"{CONVERSATIONS_DIR}_migrated"
    if not os.path.exists(CONVERSATIONS_DIR) or os.path.exists(migrated_dir_name):
        return

    json_files = [f for f in os.listdir(CONVERSATIONS_DIR) if f.endswith('.json')]
    if not json_files:
        if os.path.exists(CONVERSATIONS_DIR):
            try:
                os.rename(CONVERSATIONS_DIR, migrated_dir_name)
                print(f"Renamed empty '{CONVERSATIONS_DIR}' to '{migrated_dir_name}'.")
            except OSError as e:
                print(f"Warning: Could not rename '{CONVERSATIONS_DIR}' directory: {e}")
        return

    print("Starting migration of conversations from JSON to SQLite...")
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        
        for filename in json_files:
            filepath = os.path.join(CONVERSATIONS_DIR, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            conversation_id = data['id']
            
            cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
            if cursor.fetchone():
                continue

            created_at = data.get('created_at', datetime.now().isoformat())
            messages = data.get('messages', [])
            system_prompt = data.get('system_prompt', '')
            provider = data.get('provider', '')
            model = data.get('model', '')
            summary = data.get('summary', '')

            cursor.execute("""
            INSERT INTO conversations (id, created_at, system_prompt, provider, model, summary)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (conversation_id, created_at, system_prompt, provider, model, summary))

            for i, msg in enumerate(messages):
                cursor.execute("""
                INSERT INTO messages (conversation_id, message_index, role, content)
                VALUES (?, ?, ?, ?)
                """, (conversation_id, i, msg['role'], msg['content']))
            
            print(f"Migrated {filename} to SQLite.")
    
    print("Migration complete.")
    try:
        os.rename(CONVERSATIONS_DIR, migrated_dir_name)
        print(f"Renamed '{CONVERSATIONS_DIR}' to '{migrated_dir_name}'.")
    except OSError as e:
        print(f"Warning: Could not rename '{CONVERSATIONS_DIR}' directory: {e}")


# Initialize and migrate on first import
init_db()
migrate_json_to_sqlite()
