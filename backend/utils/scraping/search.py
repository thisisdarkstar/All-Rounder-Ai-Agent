"""
Enhanced search functionality for web, images, and files with improved error handling and rate limiting.

This module provides functionality to search the web, images, and files using multiple search engines
with proper rate limiting, retries, and error handling.
"""
import asyncio
import logging
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar, Generic, Type, cast
from urllib.parse import quote_plus, urlparse, parse_qs, urljoin

import aiohttp
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from ddgs import DDGS, AsyncDDGS
from tqdm import tqdm

from .models import SearchResult, WebPage
from .logger import get_logger
from .fetcher import download_file, fetch_webpage

# Type variable for generic return types
T = TypeVar('T')

# Configure logger
logger = get_logger(__name__)

# Default request headers to mimic a browser
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'DNT': '1',
}

@dataclass
class SearchOptions:
    """Configuration options for search operations."""
    max_retries: int = 3
    timeout: int = 30
    rate_limit_delay: float = 1.0  # seconds between requests
    max_concurrent: int = 5  # maximum concurrent requests
    user_agent: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    
    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.headers is None:
            self.headers = DEFAULT_HEADERS.copy()
        if self.user_agent:
            self.headers['User-Agent'] = self.user_agent

class RateLimiter:
    """Rate limiter for search requests."""
    
    def __init__(self, calls_per_second: float = 1.0):
        """Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum number of calls allowed per second
        """
        self.calls_per_second = calls_per_second
        self.last_call = 0.0
        self.lock = asyncio.Lock()
    
    async def __aenter__(self) -> 'RateLimiter':
        """Enter async context manager."""
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager."""
        pass
    
    async def wait(self) -> None:
        """Wait if necessary to respect the rate limit."""
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            if time_since_last < 1.0 / self.calls_per_second:
                wait_time = (1.0 / self.calls_per_second) - time_since_last
                await asyncio.sleep(wait_time)
            self.last_call = time.time()

class SearchEngine:
    """Base class for search engines."""
    
    def __init__(self, options: Optional[SearchOptions] = None):
        """Initialize search engine with options."""
        self.options = options or SearchOptions()
        self.rate_limiter = RateLimiter(1.0 / self.options.rate_limit_delay)
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Perform a search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        raise NotImplementedError("Subclasses must implement search()")
    
    async def search_images(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search for images.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of image search results
        """
        raise NotImplementedError("Subclasses must implement search_images()")
    
    async def search_files(self, query: str, file_types: Optional[List[str]] = None, 
                          max_results: int = 10) -> List[SearchResult]:
        """Search for files.
        
        Args:
            query: Search query string
            file_types: List of file extensions to search for (e.g., ['pdf', 'docx'])
            max_results: Maximum number of results to return
            
        Returns:
            List of file search results
        """
        raise NotImplementedError("Subclasses must implement search_files()")

class DuckDuckGoSearch(SearchEngine):
    """DuckDuckGo search engine implementation."""
    
    def __init__(self, options: Optional[SearchOptions] = None):
        """Initialize DuckDuckGo search engine."""
        super().__init__(options)
        self.ddgs = DDGS()
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search DuckDuckGo for web results.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            await self.rate_limiter.wait()
            results = []
            
            # Use synchronous DDGS in a thread pool
            loop = asyncio.get_event_loop()
            ddgs_results = await loop.run_in_executor(
                None, 
                lambda: list(self.ddgs.text(query, max_results=max_results))
            )
            
            for idx, result in enumerate(ddgs_results, 1):
                results.append(SearchResult(
                    title=result.get('title', '').strip(),
                    url=result.get('href', ''),
                    description=result.get('body', '').strip(),
                    source='duckduckgo',
                    rank=idx
                ))
                
            logger.info(f"DuckDuckGo search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed for '{query}': {str(e)}")
            return []
    
    async def search_images(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search DuckDuckGo for images.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of image search results
        """
        try:
            await self.rate_limiter.wait()
            results = []
            
            # Use synchronous DDGS in a thread pool
            loop = asyncio.get_event_loop()
            ddgs_results = await loop.run_in_executor(
                None, 
                lambda: list(self.ddgs.images(query, max_results=max_results))
            )
            
            for idx, result in enumerate(ddgs_results, 1):
                results.append(SearchResult(
                    title=result.get('title', '').strip(),
                    url=result.get('image', ''),
                    thumbnail=result.get('thumbnail', ''),
                    source='duckduckgo',
                    result_type='image',
                    rank=idx
                ))
                
            logger.info(f"DuckDuckGo image search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo image search failed for '{query}': {str(e)}")
            return []

class BingSearch(SearchEngine):
    """Bing search engine implementation."""
    
    BASE_URL = "https://www.bing.com"
    
    async def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Make an HTTP request with retries and rate limiting."""
        headers = self.options.headers or {}
        
        for attempt in range(self.options.max_retries + 1):
            try:
                await self.rate_limiter.wait()
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, 
                        params=params, 
                        headers=headers, 
                        timeout=self.options.timeout
                    ) as response:
                        response.raise_for_status()
                        return await response.text()
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self.options.max_retries:
                    logger.error(f"Request failed after {self.options.max_retries} attempts: {str(e)}")
                    return None
                
                # Exponential backoff
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.options.max_retries}), retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search Bing for web results.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            search_url = f"{self.BASE_URL}/search"
            params = {
                'q': query,
                'count': max_results,
                'first': 1,
                'format': 'rss'
            }
            
            html = await self._make_request(search_url, params)
            if not html:
                return []
                
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            for idx, item in enumerate(soup.select('li.b_algo'), 1):
                if idx > max_results:
                    break
                    
                title_elem = item.find('h2')
                link_elem = title_elem.find('a') if title_elem else None
                snippet_elem = item.find('p')
                
                if not (title_elem and link_elem):
                    continue
                    
                results.append(SearchResult(
                    title=title_elem.get_text(strip=True),
                    url=link_elem.get('href', ''),
                    description=snippet_elem.get_text(strip=True) if snippet_elem else '',
                    source='bing',
                    rank=idx
                ))
                
            logger.info(f"Bing search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Bing search failed for '{query}': {str(e)}")
            return []
    
    async def search_images(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search Bing for images.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of image search results
        """
        try:
            search_url = f"{self.BASE_URL}/images/search"
            params = {
                'q': query,
                'qft': '+filterui:imagesize-large',  # Filter for larger images
                'form': 'IRFLTR',
                'first': 1,
                'count': max_results
            }
            
            html = await self._make_request(search_url, params)
            if not html:
                return []
                
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            # Bing's image results are loaded via JavaScript, so we need to look for the data structure
            for idx, img in enumerate(soup.select('img.mimg'), 1):
                if idx > max_results:
                    break
                    
                img_url = img.get('src') or img.get('data-src')
                if not img_url:
                    continue
                    
                # Convert relative URLs to absolute
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif img_url.startswith('/'):
                    img_url = self.BASE_URL + img_url
                
                results.append(SearchResult(
                    title=f"Image result {idx} for '{query}'",
                    url=img_url,
                    source='bing',
                    result_type='image',
                    rank=idx
                ))
                
            logger.info(f"Bing image search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Bing image search failed for '{query}': {str(e)}")
            return []

class SearchManager:
    """Manages multiple search engines and aggregates results."""
    
    def __init__(self, engines: Optional[List[str]] = None, options: Optional[SearchOptions] = None):
        """Initialize search manager with specified engines.
        
        Args:
            engines: List of search engines to use ('duckduckgo', 'bing')
            options: Search options
        """
        self.options = options or SearchOptions()
        self.engines: Dict[str, SearchEngine] = {}
        
        # Initialize requested engines
        engines = engines or ['duckduckgo', 'bing']
        
        if 'duckduckgo' in engines:
            self.engines['duckduckgo'] = DuckDuckGoSearch(self.options)
        if 'bing' in engines:
            self.engines['bing'] = BingSearch(self.options)
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search across all configured engines and aggregate results."""
        if not self.engines:
            logger.warning("No search engines configured")
            return []
        
        tasks = [
            engine.search(query, max_results)
            for engine in self.engines.values()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten and deduplicate results
        all_results: List[SearchResult] = []
        seen_urls = set()
        
        for engine_results in results:
            if isinstance(engine_results, Exception):
                logger.error(f"Search error: {str(engine_results)}")
                continue
                
            for result in engine_results:
                if result.url and result.url not in seen_urls:
                    all_results.append(result)
                    seen_urls.add(result.url)
        
        # Sort by rank and limit results
        all_results.sort(key=lambda x: x.rank or float('inf'))
        return all_results[:max_results]
    
    async def search_images(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search for images across all configured engines."""
        if not self.engines:
            logger.warning("No search engines configured")
            return []
        
        tasks = [
            engine.search_images(query, max_results)
            for engine in self.engines.values()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten and deduplicate results
        all_results: List[SearchResult] = []
        seen_urls = set()
        
        for engine_results in results:
            if isinstance(engine_results, Exception):
                logger.error(f"Image search error: {str(engine_results)}")
                continue
                
            for result in engine_results:
                if result.url and result.url not in seen_urls:
                    all_results.append(result)
                    seen_urls.add(result.url)
        
        # Sort by rank and limit results
        all_results.sort(key=lambda x: x.rank or float('inf'))
        return all_results[:max_results]

# Convenience functions
def create_search_manager(engines: Optional[List[str]] = None, **kwargs) -> SearchManager:
    """Create a search manager with specified engines and options."""
    options = SearchOptions(**kwargs)
    return SearchManager(engines=engines, options=options)

async def search(query: str, max_results: int = 10, **kwargs) -> List[SearchResult]:
    """Search the web using default search engines."""
    async with create_search_manager(**kwargs) as manager:
        return await manager.search(query, max_results)

async def search_images(query: str, max_results: int = 10, **kwargs) -> List[SearchResult]:
    """Search for images using default search engines."""
    async with create_search_manager(**kwargs) as manager:
        return await manager.search_images(query, max_results)

# Add async context manager support to SearchManager
SearchManager.__aenter__ = lambda self: self
SearchManager.__aexit__ = lambda self, *args: None
ua = UserAgent()

# Cache for search results
SEARCH_CACHE = {}
CACHE_EXPIRY = 3600  # 1 hour cache expiry

# Search engine configurations
SEARCH_ENGINES = {
    'duckduckgo': {
        'web': 'https://html.duckduckgo.com/html/?q={query}&s={start}&dc={dc}&v=1&o=json&api=/d.js',
        'images': 'https://duckduckgo.com/i.js?q={query}&s={start}&o=json',
        'videos': 'https://duckduckgo.com/v.js?q={query}&s={start}&o=json',
        'news': 'https://duckduckgo.com/news.js?q={query}&s={start}&o=json',
        'shopping': 'https://duckduckgo.com/s.js?q={query}&s={start}&o=json',
        'files': 'https://duckduckgo.com/html/?q={query}+filetype:{filetype}&s={start}'
    },
    'bing': {
        'web': 'https://www.bing.com/search?q={query}&first={start}&count={count}',
        'images': 'https://www.bing.com/images/async?q={query}&first={start}&count={count}',
        'videos': 'https://www.bing.com/videos/asyncv2?q={query}&first={start}&count={count}',
        'news': 'https://www.bing.com/news/search?q={query}&first={start}&count={count}',
        'files': 'https://www.bing.com/search?q={query}+filetype:{filetype}&first={start}&count={count}'
    }
}

# Default headers for requests
DEFAULT_HEADERS = {
    'User-Agent': get_random_user_agent(),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Referer': 'https://www.google.com/',
    'DNT': '1'
}

class SearchEngine:
    """Base class for search engines."""
    
    def __init__(self, name: str):
        self.name = name
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search_web(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search the web."""
        raise NotImplementedError
    
    def search_images(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search for images."""
        raise NotImplementedError
    
    def search_files(self, query: str, file_types: List[str] = None, max_results: int = 10) -> List[SearchResult]:
        """Search for files of specific types."""
        raise NotImplementedError


class DuckDuckGo(SearchEngine):
    """DuckDuckGo search implementation."""
    
    def __init__(self):
        super().__init__('duckduckgo')
    
    def search_web(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search DuckDuckGo for web results."""
        try:
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(query, max_results=max_results):
                    results.append(SearchResult(
                        title=result.get('title', '').strip(),
                        url=result.get('href', ''),
                        source=self.name,
                        snippet=result.get('body', '')
                    ))
                return results
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {e}")
            return []
    
    def search_images(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search DuckDuckGo for images."""
        try:
            with DDGS() as ddgs:
                results = []
                for result in ddgs.images(query, max_results=max_results):
                    results.append(SearchResult(
                        title=result.get('title', '').strip(),
                        url=result.get('image', ''),
                        source=self.name,
                        image_url=result.get('image', ''),
                        metadata={
                            'thumbnail': result.get('thumbnail', ''),
                            'source_url': result.get('url', '')
                        }
                    ))
                return results
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo images: {e}")
            return []
    
    def search_files(self, query: str, file_types: List[str] = None, max_results: int = 10) -> List[SearchResult]:
        """Search DuckDuckGo for files of specific types."""
        if file_types:
            query += ' ' + ' OR '.join([f'filetype:{ext}' for ext in file_types])
        
        try:
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(query, max_results=max_results):
                    results.append(SearchResult(
                        title=result.get('title', '').strip(),
                        url=result.get('href', ''),
                        source=self.name,
                        file_type=Path(result.get('href', '')).suffix[1:].lower()
                    ))
                return results
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo files: {e}")
            return []


class Bing(SearchEngine):
    """Bing search implementation."""
    
    def __init__(self):
        super().__init__('bing')
        self.base_url = 'https://www.bing.com'
    
    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Get BeautifulSoup object from URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def search_web(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search Bing for web results."""
        try:
            url = f"{self.base_url}/search?q={quote_plus(query)}&count={max_results}"
            soup = self._get_soup(url)
            if not soup:
                return []
                
            results = []
            for result in soup.select('li.b_algo')[:max_results]:
                title_elem = result.find('h2')
                if not title_elem or not title_elem.a:
                    continue
                    
                title = title_elem.get_text(strip=True)
                url = title_elem.a.get('href', '')
                snippet_elem = result.find('div', class_='b_caption')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                
                results.append(SearchResult(
                    title=title,
                    url=url,
                    source=self.name,
                    snippet=snippet
                ))
            return results
        except Exception as e:
            logger.error(f"Error searching Bing: {e}")
            return []


def search_web(query: str, max_results: int = 10, engines: List[str] = None) -> List[SearchResult]:
    """Search multiple search engines for web results."""
    engines = engines or ['duckduckgo', 'bing']
    results = []
    
    if 'duckduckgo' in engines:
        results.extend(DuckDuckGo().search_web(query, max_results))
    if 'bing' in engines:
        results.extend(Bing().search_web(query, max_results))
    
    return results


def search_images(query: str, max_results: int = 10, engines: List[str] = None) -> List[SearchResult]:
    """Search for images across multiple search engines."""
    engines = engines or ['duckduckgo']  # Bing image search requires API key
    results = []
    
    if 'duckduckgo' in engines:
        results.extend(DuckDuckGo().search_images(query, max_results))
    
    return results


def search_files(query: str, file_types: List[str] = None, max_results: int = 10, 
                engines: List[str] = None) -> List[SearchResult]:
    """Search for files across multiple search engines."""
    engines = engines or ['duckduckgo', 'bing']
    results = []
    
    if 'duckduckgo' in engines:
        results.extend(DuckDuckGo().search_files(query, file_types, max_results))
    
    return results
