from __future__ import annotations
from typing import Dict, List, Any, Optional, TypedDict
from pathlib import Path
import json
import logging
import os
from dataclasses import dataclass
from openai import OpenAI, APIError, APITimeoutError
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
    
    # Prepare messages with system prompt and few-shot examples
    messages: List[ChatMessage] = [
        {"role": "system", "content": preamble},
        *FEWSHOTS,
        {"role": "user", "content": prompt}
    ]
    
    # Make API request with retries
    for attempt in range(CONFIG["max_retries"]):
        try:
            result = client.chat.completions.create(
                model=CONFIG["model_name"],
                messages=messages,
                temperature=0.2,
                max_tokens=100,
            )
            
            # Process the response
            content = result.choices[0].message.content
            if not content:
                logger.warning("Empty response from LLM")
                return ["general"]
                
            # Parse and validate response
            output = [s.strip() for s in content.replace("\n", "").split(",") if s.strip()]
            output = normalize_response(output)
            
            # Filter valid responses
            valid_responses = [
                task for task in output 
                if any(task.startswith(known) for known in KNOWN_TYPES)
            ]
            
            # Log the classification result
            if CONFIG["log_enabled"]:
                save_chat_log([{"role": "user", "content": prompt}])
                
            return valid_responses or ["general"]
            
        except APITimeoutError:
            if attempt == CONFIG["max_retries"] - 1:
                logger.error("API request timed out after retries")
                raise RuntimeError("Service unavailable. Please try again later.")
            logger.warning(f"API timeout, retrying... (attempt {attempt + 1}/{CONFIG['max_retries']})")
            
        except APIError as e:
            logger.error(f"API error: {e}")
            raise RuntimeError(f"Error communicating with the AI service: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error in classify_query: {e}")
            if attempt == CONFIG["max_retries"] - 1:
                raise RuntimeError("An unexpected error occurred. Please try again.")
    
    # If all retries failed
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
