import random
import difflib

def validate_chat_shape(chat):
    """
    Validates that the chat JSON is well-formed:
    - Has "messages" key which is a list.
    - First message has role "system".
    - Roles alternate user, assistant, user, assistant...
    Returns: (is_valid, reason)
    """
    if not isinstance(chat, dict):
        return False, "Chat is not a JSON object."
    
    if "messages" not in chat:
        return False, "Missing 'messages' key."
        
    messages = chat["messages"]
    if not isinstance(messages, list):
        return False, "'messages' is not a list."
        
    if len(messages) < 3:
        return False, f"Chat has too few messages ({len(messages)}), needs at least 3 (system, user, assistant)."
        
    # Check first role
    if messages[0].get("role") != "system":
        return False, f"First message role is '{messages[0].get('role')}', expected 'system'."
        
    # Check alternating user and assistant
    for i in range(1, len(messages)):
        role = messages[i].get("role")
        content = messages[i].get("content")
        
        if not role or not content:
            return False, f"Message {i} is missing 'role' or 'content'."
            
        expected_role = "user" if i % 2 == 1 else "assistant"
        if role != expected_role:
            return False, f"Message {i} has role '{role}', expected '{expected_role}' (alternating)."
            
    return True, "Valid shape."

def get_chat_text(chat):
    """
    Concatenates all user and assistant content into a single string for comparison.
    """
    return " ".join([m["content"] for m in chat["messages"] if m["role"] in ("user", "assistant")])

def estimate_tokens(text):
    """
    Simple heuristic to estimate tokens.
    For mixed English/Hindi/Hinglish:
    - English: ~1.3 tokens per word
    - Hinglish: ~1.4 tokens per word
    - Devanagari Hindi: ~2.5 tokens per word (as characters are often multi-token)
    We will scan if it has devanagari characters.
    """
    has_devanagari = bool(has_devanagari_chars(text))
    words = len(text.split())
    if has_devanagari:
        # Devanagari character count is a better proxy
        return int(len(text) * 0.8) # roughly 0.8 tokens per character in Devanagari
    else:
        return int(words * 1.3)

def has_devanagari_chars(text):
    return any('\u0900' <= char <= '\u097f' for char in text)

def analyze_chat_length(chat):
    """
    Counts words and estimates tokens for the chat.
    """
    system_text = " ".join([m["content"] for m in chat["messages"] if m["role"] == "system"])
    chat_text = get_chat_text(chat)
    
    system_words = len(system_text.split())
    chat_words = len(chat_text.split())
    total_words = system_words + chat_words
    
    system_tokens = estimate_tokens(system_text)
    chat_tokens = estimate_tokens(chat_text)
    total_tokens = system_tokens + chat_tokens
    
    return {
        "system_words": system_words,
        "chat_words": chat_words,
        "total_words": total_words,
        "system_tokens": system_tokens,
        "chat_tokens": chat_tokens,
        "total_tokens": total_tokens
    }

def find_near_duplicates(chats, threshold=0.85):
    """
    Finds near-duplicate chats.
    Returns: list of dicts with (index_a, index_b, similarity)
    """
    duplicates = []
    chat_texts = [get_chat_text(chat) for chat in chats]
    
    for i in range(len(chat_texts)):
        for j in range(i + 1, len(chat_texts)):
            # Fast length check first to avoid heavy sequence matching if lengths differ vastly
            len_i, len_j = len(chat_texts[i]), len(chat_texts[j])
            if len_i == 0 or len_j == 0:
                continue
            ratio_len = min(len_i, len_j) / max(len_i, len_j)
            if ratio_len < threshold - 0.1:
                continue
                
            similarity = difflib.SequenceMatcher(None, chat_texts[i], chat_texts[j]).ratio()
            if similarity >= threshold:
                duplicates.append({
                    "chat_a": i + 1,
                    "chat_b": j + 1,
                    "similarity": similarity
                })
    return duplicates

def split_dataset(chats, train_ratio=0.8, seed=42):
    """
    Splits the chats list into train and test sets.
    """
    # Seed for reproducibility
    random.seed(seed)
    shuffled = chats.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_set = shuffled[:split_idx]
    test_set = shuffled[split_idx:]
    
    return train_set, test_set
