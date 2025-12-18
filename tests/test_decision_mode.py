import pytest
from backend.Model import classify_query, KNOWN_TYPES

# Test cases for each known type and edge cases
TEST_CASES = [
    # General queries
    ("how are you", ["general how are you"]),
    ("do you like pizza", ["general do you like pizza"]),
    ("explain python", ["general explain python"]),  # Mapped from 'content python' via CANONICAL_MAP
    ("who was akbar", ["general who was akbar"]),
    
    # Realtime information
    ("who is elon musk", ["realtime who is elon musk"]),
    ("who is indian prime minister", ["realtime who is indian prime minister"]),
    ("what is the news today", ["realtime what is the news today"]),
    
    # Application control
    ("open whatsapp", ["open whatsapp"]),
    ("close notepad", ["close notepad"]),
    ("open chrome and close firefox", ["open chrome", "close firefox"]),
    ("open telegram, open facebook, close whatsapp", ["open telegram", "open facebook", "close whatsapp"]),
    
    # Media control
    ("play despacito", ["play despacito"]),
    ("play let it be and open spotify", ["play let it be", "open spotify"]),
    ("generate image of a lion", ["generate image of a lion"]),
    
    # System actions
    ("set a reminder for meeting at 2pm", ["reminder 2pm meeting"]),
    ("mute system volume", ["system mute"]),
    ("exit", ["system exit"]),  # Changed from general exit to system exit
    
    # Search operations
    ("search python in google", ["google search python"]),
    ("search news on youtube", ["youtube search news"]),
    
    # Edge cases and error handling
    ("", ["general"]),
    ("   ", ["general"]),  # Whitespace only
    ("do something", ["general do something"]),
    ("can you check this", ["general can you check this"]),
]

@pytest.mark.parametrize("query,expected_categories", TEST_CASES)
def test_classify_query(query, expected_categories):
    """Test that classify_query returns the expected categories for various inputs."""
    result = classify_query(query)
    
    # Check that the result is a non-empty list
    assert isinstance(result, list)
    assert len(result) > 0
    
    # Check that all returned categories are in the known types
    for category in result:
        assert any(category.startswith(known) for known in KNOWN_TYPES), \
            f"Category '{category}' not in KNOWN_TYPES"
    
    # Check that all expected categories are in the result
    for expected in expected_categories:
        assert any(expected in category for category in result), \
            f"Expected category containing '{expected}' not found in {result}"

if __name__ == "__main__":
    pytest.main()
