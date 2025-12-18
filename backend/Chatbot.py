import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from openai import OpenAI, APIError, APITimeoutError
from dotenv import load_dotenv
from utils.prompts import FRIDAY_PERSONA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "model_name": os.getenv("LLM_MODEL", "qwen2.5-1.5b-instruct"),
    "endpoint": os.getenv("LLM_ENDPOINT", "http://localhost:1234/v1"),
    "api_key": os.getenv("LLM_API_KEY", "lm-studio"),
    "chat_log_path": Path("backend/logs/Chatbot_history.json"),
    "memory_window": 10,
    "max_tokens": 1500,
    "temperature": 0.5,
    "request_timeout": 30.0,
}

# Initialize OpenAI client
client = OpenAI(
    base_url=CONFIG["endpoint"],
    api_key=CONFIG["api_key"],
    timeout=CONFIG["request_timeout"]
)


def load_chat_log() -> List[Dict[str, str]]:
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


def save_chat_log(log: List[Dict[str, str]]) -> None:
    """Save chat history to JSON file with error handling."""
    try:
        CONFIG["chat_log_path"].parent.mkdir(parents=True, exist_ok=True)
        with CONFIG["chat_log_path"].open('w', encoding='utf-8') as f:
            json.dump(log, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving chat log: {e}")


def get_chat_response(messages: List[Dict[str, str]]) -> str:
    """Get response from the language model with error handling and retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=CONFIG["model_name"],
                messages=messages,
                temperature=CONFIG["temperature"],
                max_tokens=CONFIG["max_tokens"],
            )
            return response.choices[0].message.content.strip()
            
        except APITimeoutError:
            if attempt == max_retries - 1:
                logger.error("API request timed out after retries")
                return "I'm having trouble connecting to the AI service. Please try again later."
            logger.warning(f"API timeout, retrying... (attempt {attempt + 1}/{max_retries})")
            
        except APIError as e:
            logger.error(f"API error: {e}")
            return "I'm experiencing technical difficulties. Please try again in a moment."
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return "An unexpected error occurred. Please try again."


def chat_session() -> None:
    """Main chat session loop."""
    print("Friday: Hi! I'm Friday, your personal assistant. How may I help you today?")
    
    # Initialize message history
    messages = [{"role": "system", "content": FRIDAY_PERSONA}]
    history = load_chat_log()
    
    # Load recent conversation history as context
    for entry in history[-CONFIG["memory_window"]:]:
        messages.extend([
            {"role": "user", "content": entry["user"]},
            {"role": "assistant", "content": entry["assistant"]}
        ])
    
    try:
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                if not user_input:
                    print("Please enter a message.")
                    continue
                
                # Add user message to conversation
                messages.append({"role": "user", "content": user_input})
                
                # Get AI response
                print("\nFriday: ", end="", flush=True)
                answer = get_chat_response(messages)
                
                # Stream the response for better UX
                for chunk in answer.split(' '):
                    print(chunk, end=' ', flush=True)
                print()  # Add newline after response
                
                # Update conversation history
                messages.append({"role": "assistant", "content": answer})
                history.append({
                    "timestamp": datetime.now().isoformat(),
                    "user": user_input,
                    "assistant": answer,
                })
                
                # Save conversation history
                save_chat_log(history)
                
                # Enforce message window to prevent context overflow
                if len(messages) > CONFIG["memory_window"] * 2 + 1:  # *2 for user/assistant pairs, +1 for system message
                    messages = [messages[0]] + messages[-(CONFIG["memory_window"] * 2):]
            
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print("Sorry, I encountered an error. Let's try that again.")
                continue
                
    except KeyboardInterrupt:
        print("\n\nFriday: Goodbye! Have a great day! 👋")
    except Exception as e:
        logger.critical(f"Fatal error in chat session: {e}")
        print("\nA critical error occurred. Please restart the application.")


if __name__ == "__main__":
    chat_session()
