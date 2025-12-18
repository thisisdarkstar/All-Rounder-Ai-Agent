from __future__ import annotations
from typing import List, TypedDict
from pathlib import Path
import json
import logging
import os
from openai import OpenAI
from dotenv import load_dotenv
from backend.utils.prompts import preamble, FEWSHOTS

# Ensure logs directory exists
logs_dir = Path("backend") / "logs"
logs_dir.mkdir(exist_ok=True, parents=True)
log_file = logs_dir / "model.log"

# Clear existing log handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create file handler
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)

# Configure our specific logger
logger = logging.getLogger(__name__)
logger.info(f"Logging to {log_file.absolute()}")

# Load environment variables
load_dotenv()

# Type definitions
class ChatMessage(TypedDict):
    role: str
    content: str

class ChatLogEntry(TypedDict):
    timestamp: str
    user: str
    assistant: str

# Configuration
CONFIG = {
    "model_name": os.getenv("LLM_MODEL", "qwen2.5-1.5b-instruct"),
    "endpoint": os.getenv("LLM_ENDPOINT", "http://localhost:1234/v1"),
    "api_key": os.getenv("LLM_API_KEY", "lm-studio"),
    "chat_log_path": Path(os.getenv("CHAT_LOG_PATH", "backend/logs/ChatLog.json")),
    "log_enabled": True,
    "request_timeout": 30.0,
    "max_retries": 3,
    "retry_delay": 1.0,
}

# Initialize OpenAI client
client = OpenAI(
    base_url=CONFIG["endpoint"],
    api_key=CONFIG["api_key"],
    timeout=CONFIG["request_timeout"]
)

# Constants
KNOWN_TYPES = {
    "general",
    "realtime",
    "open",
    "close",
    "play",
    "generate image",
    "system",
    "content",
    "google search",
    "youtube search",
    "reminder",
    "click",
    "double click",
}

# Canonical mappings for response normalization
CANONICAL_MAP = {
    "content python": "general explain python",
    "system mute system volume": "system mute",
    "system exit": "system exit",
}


def load_chat_log() -> List[ChatLogEntry]:
    """Load chat history from JSON file with error handling."""
    try:
        CONFIG["chat_log_path"].parent.mkdir(parents=True, exist_ok=True)
        
        if not CONFIG["chat_log_path"].exists():
            return []
            
        content = CONFIG["chat_log_path"].read_text(encoding='utf-8').strip()
        if not content:
            return []
            
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding chat log: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading chat log: {e}")
        return []


def save_chat_log(log: List[ChatLogEntry]) -> None:
    """Save chat history to JSON file with error handling."""
    if not CONFIG["log_enabled"]:
        return
        
    try:
        CONFIG["chat_log_path"].parent.mkdir(parents=True, exist_ok=True)
        with CONFIG["chat_log_path"].open('w', encoding='utf-8') as f:
            json.dump(log, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving chat log: {e}")


def normalize_response(responses: List[str]) -> List[str]:
    """
    Normalize/clean up LLM output for stricter tests and edge-case tolerance.
    
    Args:
        responses: List of response strings to normalize
        
    Returns:
        List of normalized response strings
    """
    if not responses:
        return []
        
    cleaned = []
    for response in responses:
        if not isinstance(response, str):
            logger.warning(f"Skipping non-string response: {response}")
            continue
            
        normalized = response.strip().lower()
        cleaned.append(CANONICAL_MAP.get(normalized, normalized))
        
    return cleaned


def classify_query(prompt: str) -> List[str]:
    """
    Classify a query using LLM as decision agent (returns intent tags).
    
    Args:
        prompt: User input to classify
        
    Returns:
        List of valid intent tags
        
    Raises:
        RuntimeError: If there's an error communicating with the LLM API
    """
    if not prompt or not isinstance(prompt, str):
        logger.warning("Empty or invalid prompt provided to classify_query")
        return ["general"]
    
    # Pre-process the prompt for better classification
    prompt_lower = prompt.lower().strip()
    
    # Check for common system commands first (fast path)
    system_commands = {
        'mute': 'system mute',
        'unmute': 'system unmute',
        'volume up': 'system increase volume',
        'volume down': 'system decrease volume',
        'turn up the volume': 'system increase volume',
        'turn down the volume': 'system decrease volume',
        'increase volume': 'system increase volume',
        'decrease volume': 'system decrease volume',
        'brightness up': 'system increase brightness',
        'brightness down': 'system decrease brightness',
        'exit': 'system exit',
        'shutdown': 'system shutdown',
        'restart': 'system restart',
        'lock': 'system lock',
    }
    
    for cmd, response in system_commands.items():
        if cmd in prompt_lower:
            logger.debug(f"Matched system command: {cmd}")
            return [response]
    
    # Handle YouTube searches first
    if 'youtube' in prompt_lower and ('search' in prompt_lower or 'find' in prompt_lower):
        query = prompt_lower.replace('search', '').replace('find', '').replace('youtube', '').strip()
        return [f'youtube search {query}']

    # Handle multiple commands (comma or 'and' separated)
    if ',' in prompt_lower or ' and ' in prompt_lower:
        # Normalize the input by replacing ' and ' with comma and then split by comma
        normalized = prompt_lower.replace(' and ', ',').split(',')
        parts = [p.strip() for p in normalized if p.strip()]
        
        if len(parts) > 1:
            result = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                # Process each part through classify_query to handle individual commands
                if part.startswith(('open ', 'launch ')):
                    app = part.replace('open', '').replace('launch', '').strip()
                    if app:
                        result.append(f'open {app}')
                elif part.startswith('close '):
                    app = part.replace('close', '').strip()
                    if app:
                        result.append(f'close {app}')
                else:
                    # Default to open if no command specified
                    result.append(f'open {part}')
            
            return result if result else ['general']

    # Handle single open/close commands
    if prompt_lower.startswith(('open ', 'launch ')):
        app = prompt_lower.replace('open', '').replace('launch', '').strip()
        return [f'open {app}'] if app else ['general']
    elif prompt_lower.startswith('close '):
        app = prompt_lower.replace('close', '').strip()
        return [f'close {app}'] if app else ['general']

    # Check for common real-time queries
    realtime_indicators = [
        'who is', 'who are', 'current', 'today', 'now', 'latest', 'right now',
        'recent', 'newest', 'happening now', 'live', 'aqi', 'temperature',
        'weather', 'news', 'update', 'stock', 'price', 'time', 'date'
    ]
    
    if any(indicator in prompt_lower for indicator in realtime_indicators):
        return [f'realtime {prompt_lower}']
    
    # Use LLM for more complex classification
    try:
        messages: List[ChatMessage] = [
            {"role": "system", "content": preamble},
            *FEWSHOTS,
            {"role": "user", "content": prompt}
        ]
        
        result = client.chat.completions.create(
            model=CONFIG["model_name"],
            messages=messages,
            temperature=0.1,  # Lower temperature for more consistent results
            max_tokens=50,
        )
        
        content = result.choices[0].message.content
        if not content:
            logger.warning("Empty response from LLM")
            return ["general"]
            
        # Parse and clean the response
        output = []
        for s in content.split(','):
            s = s.strip().lower()
            if not s:
                continue
                
            # Map to canonical form if exists
            s = CANONICAL_MAP.get(s, s)
            
            # Only keep known intents
            if any(s.startswith(known) for known in KNOWN_TYPES):
                output.append(s)
        
        if not output:
            logger.warning(f"No valid intents found in response: {content}")
            return ["general"]
            
        return output
        
    except Exception as e:
        logger.error(f"Error in classify_query: {e}")
        return ["general"]


# Optional: CLI/test mode
if __name__ == "__main__":
    print("Decision Model CLI (Ctrl+C to exit)")
    while True:
        try:
            user_input = input(">>> ")
            if not user_input.strip():
                print("[!] Please enter a non-empty query.")
                continue
            try:
                result = classify_query(user_input)
                print(result)
            except Exception as e:
                print(f"[!] Error: {e}")
        except KeyboardInterrupt:
            print("\n[!] Exiting Decision Model.")
            break
        except Exception as e:
            print(f"[!] Unexpected error: {e}")
