"""
Web Scraping Utilities

This package provides utilities for web scraping, including:
- Web search (DuckDuckGo, Bing)
- Image search and download
- File search and download
- Web page content extraction
"""

from .search import search_web, search_images, search_files
from .fetcher import fetch_webpage, download_file, download_image
from .models import SearchResult, WebPage

__all__ = [
    'search_web',
    'search_images',
    'search_files',
    'fetch_webpage',
    'download_file',
    'download_image',
    'SearchResult',
    'WebPage'
]
