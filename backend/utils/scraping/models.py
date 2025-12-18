"""
Data models for the scraping package.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class SearchResult:
    """Represents a search result from a search engine."""
    title: str
    url: str
    source: str  # e.g., 'duckduckgo', 'bing'
    snippet: Optional[str] = None
    image_url: Optional[str] = None
    file_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class WebPage:
    """Represents a downloaded web page with its content."""
    url: str
    title: str
    content: str
    status_code: int
    headers: dict
    images: List[Dict[str, str]] = None
    links: List[Dict[str, str]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.images = self.images or []
        self.links = self.links or []
        self.metadata = self.metadata or {}
