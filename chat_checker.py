import os
import sys
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Force UTF-8 encoding for standard output to support Devanagari characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.utils import validate_chat_shape, analyze_chat_length, find_near_duplicates, split_dataset
from src.safety import audit_chat, SAFETY_RULES

# Load env variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Vedaz AI Astrologer Chat Checker")
    parser.add_argument("--input", type=str, default="data/vedaz_astrologer_finetune.jsonl", help="Path to input JSONL file")
    parser.add_argument("--train-out", type=str, default="data/train.jsonl", help="Path to output train JSONL file")
    parser.add_argument("--test-out", type=str, default="data/test.jsonl", help="Path to output test JSONL file")
    parser.add_argument("--similarity", type=float, default=0.85, help="Near-duplicate similarity threshold (0.0 to 1.0)")
    parser.add_argument("--llm", action="store_true", help="Enable LLM-based safety audit (requires .env configuration)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return
        
    print(f"\n==================================================")
    print(f"   VEDAZ CHAT CHECKER - REPORT")
    print(f"==================================================\n")
    print(f"Analyzing file: {args.input}")
    
    # Initialize OpenAI Client if LLM auditing is requested
    client = None
    model_name = None
    if args.llm:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        if not api_key:
            print("Warning: LLM audit requested but OPENAI_API_KEY is not set in environment. Falling back to keyword heuristics.")
        else:
            client_args = {"api_key": api_key}
            if base_url:
                client_args["base_url"] = base_url
            client = OpenAI(**client_args)
            print(f"LLM Audit enabled. Model: {model_name}")
            if base_url:
                print(f"Using Custom Base URL: {base_url}")
    else:
        print("LLM Audit disabled. Using keyword heuristics only.")

    raw_chats = []
    malformed_lines = []
    
    # 1. Read and parse lines
    with open(args.input, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chat = json.loads(line)
                raw_chats.append((idx, chat))
            except json.JSONDecodeError as e:
                malformed_lines.append((idx, f"Invalid JSON format: {str(e)}"))
                
    total_lines = len(raw_chats) + len(malformed_lines)
    print(f"Read {total_lines} total lines from file. ({len(raw_chats)} parsed JSON objects, {len(malformed_lines)} parsing errors)\n")
    
    # 2. Validate shapes and analyze chats
    valid_chats = []
    invalid_chats = []
    
    for original_idx, chat in raw_chats:
        is_shape_valid, shape_reason = validate_chat_shape(chat)
        if not is_shape_valid:
            invalid_chats.append({
                "index": original_idx,
                "chat": chat,
                "reason": shape_reason
            })
        else:
            # Analyze lengths
            lengths = analyze_chat_length(chat)
            
            # Check safety
            is_safe, violated_rules, safety_reason = audit_chat(chat["messages"], client, model_name)
            
            valid_chats.append({
                "index": original_idx,
                "chat": chat,
                "lengths": lengths,
                "is_safe": is_safe,
                "violated_rules": violated_rules,
                "safety_reason": safety_reason,
                "is_duplicate": False,
                "duplicate_info": None
            })
            
    # 3. Find near-duplicates among valid chats
    only_chats = [item["chat"] for item in valid_chats]
    duplicates_list = find_near_duplicates(only_chats, threshold=args.similarity)
    
    # Map back duplicates to valid_chats list
    # duplicates_list has 1-based indices relative to the 'only_chats' list
    for dup in duplicates_list:
        idx_a = dup["chat_a"] - 1 # 0-indexed in valid_chats
        idx_b = dup["chat_b"] - 1
        
        # Mark the second one (b) as a duplicate of the first (a)
        valid_chats[idx_b]["is_duplicate"] = True
        valid_chats[idx_b]["duplicate_info"] = f"Near-duplicate of Chat #{valid_chats[idx_a]['index']} (similarity: {dup['similarity']:.2%})"

    # Count safety flags
    unsafe_chats = [c for c in valid_chats if not c["is_safe"]]
    safe_unique_chats = [c for c in valid_chats if c["is_safe"] and not c["is_duplicate"]]
    
    # 4. Print detailed reports
    try:
        from tabulate import tabulate
        headers = ["Line #", "Shape", "Words", "Est. Tokens", "Safety", "Duplicates / Errors"]
        table_data = []
        
        # Add malformed errors
        for idx, err in malformed_lines:
            table_data.append([idx, "FAIL", "-", "-", "Skipped", f"JSON Error: {err}"])
            
        # Add invalid shape errors
        for item in invalid_chats:
            table_data.append([item["index"], "FAIL", "-", "-", "Skipped", f"Shape Error: {item['reason']}"])
            
        # Add valid chats
        for item in valid_chats:
            shape_str = "PASS"
            words = item["lengths"]["total_words"]
            tokens = item["lengths"]["total_tokens"]
            
            safety_str = "SAFE"
            if not item["is_safe"]:
                rules_str = ",".join([f"Rule {r}" for r in item["violated_rules"]])
                safety_str = f"FLAG ({rules_str})"
                
            dup_str = "-"
            if item["is_duplicate"]:
                dup_str = f"DUP: {item['duplicate_info']}"
                
            table_data.append([item["index"], shape_str, words, tokens, safety_str, dup_str])
            
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except ImportError:
        # Fallback print if tabulate not installed
        print("Line # | Shape | Words | Est. Tokens | Safety | Duplicates/Errors")
        print("-" * 80)
        for idx, err in malformed_lines:
            print(f"{idx} | FAIL | - | - | Skipped | JSON Error: {err}")
        for item in invalid_chats:
            print(f"{item['index']} | FAIL | - | - | Skipped | Shape Error: {item['reason']}")
        for item in valid_chats:
            safety_str = "SAFE" if item["is_safe"] else f"FLAGGED ({item['violated_rules']})"
            dup_str = item["duplicate_info"] if item["is_duplicate"] else "-"
            print(f"{item['index']} | PASS | {item['lengths']['total_words']} | {item['lengths']['total_tokens']} | {safety_str} | {dup_str}")

    # Print safety violations details
    if unsafe_chats:
        print(f"\n==================================================")
        print(f"   SAFETY VIOLATIONS DETAILS")
        print(f"==================================================")
        for c in unsafe_chats:
            print(f"\nLine #{c['index']}:")
            print(f"  Violated Rules: {c['violated_rules']}")
            for r in c['violated_rules']:
                print(f"    - {SAFETY_RULES.get(r, 'Unknown Rule')}")
            print(f"  Audit Reason: {c['safety_reason']}")

    # Print summary statistics
    print(f"\n==================================================")
    print(f"   SUMMARY STATISTICS")
    print(f"==================================================")
    print(f"  Total Lines Audited:          {total_lines}")
    print(f"  Malformed Lines (JSON):       {len(malformed_lines)}")
    print(f"  Invalid Shape Chats:          {len(invalid_chats)}")
    print(f"  Valid Chats:                  {len(valid_chats)}")
    print(f"  Unsafe / Flagged Chats:       {len(unsafe_chats)}")
    print(f"  Near-Duplicate Chats:         {len(duplicates_list)}")
    print(f"  Safe & Unique Chats:          {len(safe_unique_chats)}")
    print(f"==================================================\n")

    # 5. Split and save dataset
    if len(safe_unique_chats) == 0:
        print("Warning: No safe and unique chats found to split.")
        return
        
    train_set, test_set = split_dataset([c["chat"] for c in safe_unique_chats], train_ratio=0.8, seed=42)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.test_out), exist_ok=True)
    
    with open(args.train_out, "w", encoding="utf-8") as f:
        for chat in train_set:
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")
            
    with open(args.test_out, "w", encoding="utf-8") as f:
        for chat in test_set:
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")
            
    print(f"Dataset split complete:")
    print(f"  - Saved {len(train_set)} chats to '{args.train_out}' (Training set)")
    print(f"  - Saved {len(test_set)} chats to '{args.test_out}' (Test set)")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()
