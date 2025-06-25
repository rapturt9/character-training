
import json
import os
from database import get_message_by_index_from_db, load_conversation_from_db, migrate_json_to_sqlite

# This is a test script to verify migration and message retrieval

# 1. Define the conversation to test
# Using a known conversation ID from your workspace
TEST_CONVERSATION_ID = "0f1622b9-24cb-497e-bba7-89618f3e42db"
JSON_FILE_PATH = os.path.join("conversations", f"conversation_{TEST_CONVERSATION_ID}.json")

def run_test():
    print("--- Running Migration and Search Test ---")

    # Ensure migration is run
    print("Running migration function...")
    migrate_json_to_sqlite()

    # 2. Read the original data from the JSON file
    try:
        with open(JSON_FILE_PATH, 'r') as f:
            original_data = json.load(f)
        print(f"Successfully read original data from {JSON_FILE_PATH}")
    except FileNotFoundError:
        print(f"ERROR: JSON file not found at {JSON_FILE_PATH}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {JSON_FILE_PATH}")
        return

    original_messages = original_data.get("messages", [])
    if not original_messages:
        print("No messages found in the original JSON file.")
        return

    print(f"Found {len(original_messages)} messages in the JSON file.")

    # 3. Load the conversation from the database
    print(f"Attempting to load conversation '{TEST_CONVERSATION_ID}' from the database...")
    db_conversation = load_conversation_from_db(TEST_CONVERSATION_ID)

    if not db_conversation:
        print("ERROR: Conversation not found in the database. Migration might have failed.")
        return

    print("Successfully loaded conversation from the database.")
    db_messages = db_conversation.get("messages", [])

    # 4. Compare the messages from JSON and DB
    if len(original_messages) != len(db_messages):
        print(f"ERROR: Message count mismatch! JSON has {len(original_messages)}, DB has {len(db_messages)}.")
        return

    print("Message count matches between JSON and database.")
    # Simple content check
    if original_messages[0]['content'] == db_messages[0]['content']:
        print("Content of the first message matches.")
    else:
        print("ERROR: Content of the first message does not match.")
        return

    # 5. Test get_message_by_index_from_db
    TEST_MESSAGE_INDEX = 0
    print(f"Testing get_message_by_index_from_db for message index {TEST_MESSAGE_INDEX}...")
    db_message = get_message_by_index_from_db(TEST_CONVERSATION_ID, TEST_MESSAGE_INDEX)

    if not db_message:
        print(f"ERROR: get_message_by_index_from_db failed to retrieve message.")
        return

    print("Successfully retrieved message by index from the database.")

    # 6. Compare the retrieved message with the original
    original_message = original_messages[TEST_MESSAGE_INDEX]

    if db_message['role'] == original_message['role'] and db_message['content'] == original_message['content']:
        print("\nSUCCESS: The retrieved message matches the original message from the JSON file.")
    else:
        print("\nERROR: The retrieved message does not match the original.")
        print(f"Original: {original_message}")
        print(f"From DB: {db_message}")

    print("--- Test Finished ---")

if __name__ == "__main__":
    run_test()
