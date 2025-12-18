"""
Enhanced utility functions for web scraping with improved performance and reliability.
"""
import asyncio
import aiohttp
import time
import random
import re
import logging
import json
import socket
import ssl
from urllib.parse import urlparse, urljoin, urlunparse, parse_qs, urlencode
from typing import Optional, Any, List, Tuple
from dataclasses import asdict, is_dataclass
from functools import wraps
from contextlib import asynccontextmanager
import backoff
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import tldextract

logger = logging.getLogger(__name__)

# Global session for connection reuse
_SESSION = None

class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass

class RateLimiter:
    """Enhanced rate limiting with async support and backoff strategies."""
    
    def __init__(self, max_requests: int = 5, per_seconds: float = 1.0, 
                 jitter: Tuple[float, float] = (0.5, 1.5)):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.jitter = jitter
        self.timestamps = []
        self._lock = asyncio.Lock()
    
    async def wait(self) -> None:
        """Wait if rate limit is reached with jitter for more natural behavior."""
        async with self._lock:
            now = time.monotonic()
            
            # Remove old timestamps outside the rate limit window
            self.timestamps = [t for t in self.timestamps 
                             if now - t < self.per_seconds]
            
            if len(self.timestamps) >= self.max_requests:
                # Calculate sleep time with jitter
                sleep_time = (self.per_seconds - (now - self.timestamps[0]))
                jitter = random.uniform(*self.jitter)
                sleep_time = max(0, sleep_time * jitter)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                # Update timestamps after sleeping
                now = time.monotonic()
                self.timestamps = [t for t in self.timestamps 
                                 if now - t < self.per_seconds]
            
            self.timestamps.append(now)

def rate_limited(max_requests: int = 5, per_seconds: float = 1.0):
    """Decorator to rate limit function calls."""
    rate_limiter = RateLimiter(max_requests, per_seconds)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await rate_limiter.wait()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def get_random_user_agent() -> str:
    """Get a random user agent for requests with fallback."""
    try:
        ua = UserAgent()
        return ua.random
    except Exception as e:
        logger.warning(f"Failed to get random user agent: {e}")
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )

def normalize_url(url: str, base_url: str = None) -> str:
    """
    Normalize URL by:
    1. Converting to lowercase
    2. Removing default ports
    3. Removing fragments
    4. Sorting query parameters
    5. Removing tracking parameters
    """
    if not url:
        return ""
    
    # Handle relative URLs
    if base_url and not url.startswith(('http://', 'https://')):
        url = urljoin(base_url, url)
    
    try:
        parsed = urlparse(url.lower())
        
        # Remove default ports
        netloc = parsed.netloc
        if ':' in netloc:
            host, port = netloc.split(':', 1)
            if (parsed.scheme == 'http' and port == '80') or \
               (parsed.scheme == 'https' and port == '443'):
                netloc = host
        
        # Sort query parameters and remove tracking params
        query_parts = []
        if parsed.query:
            params = parse_qs(parsed.query, keep_blank_values=True)
            # Filter out common tracking parameters
            tracking_params = {
                'utm_', 'ref_', 'source', 'campaign', 'fbclid', 'gclid', 
                'gclsrc', 'dclid', 'msclkid', 'mc_cid', 'mc_eid', '_ga'
            }
            
            clean_params = {}
            for k, v in params.items():
                if not any(tp in k.lower() for tp in tracking_params):
                    clean_params[k] = v
            
            if clean_params:
                query_parts.append(urlencode(clean_params, doseq=True))
        
        # Rebuild URL
        return urlunparse((
            parsed.scheme,
            netloc,
            parsed.path.rstrip('/') or '/',  # Normalize empty path to '/'
            parsed.params,
            '&'.join(query_parts) if query_parts else '',
            ''  # Remove fragment
        ))
    except Exception as e:
        logger.warning(f"Failed to normalize URL {url}: {e}")
        return url

def get_domain(url: str) -> str:
    """Extract domain from URL using tldextract for better accuracy."""
    try:
        extracted = tldextract.extract(url)
        if not extracted.domain or not extracted.suffix:
            raise ValueError("Invalid domain")
        return f"{extracted.domain}.{extracted.suffix}"
    except Exception as e:
        logger.warning(f"Failed to extract domain from {url}: {e}")
        parsed = urlparse(url)
        return parsed.netloc or ''

def is_valid_url(url: str, require_https: bool = True) -> bool:
    """
    Validate URL with additional checks:
    - Valid scheme (http/https)
    - Valid netloc
    - Valid TLD
    - Optional HTTPS requirement
    """
    if not url or not isinstance(url, str):
        return False
    
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
            
        if require_https and result.scheme != 'https':
            return False
            
        # Validate TLD
        extracted = tldextract.extract(url)
        if not extracted.suffix or '.' not in extracted.suffix:
            return False
            
        # Validate domain characters
        if not re.match(r'^[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,}$', 
                       extracted.registered_domain.lower()):
            return False
            
        return True
    except (ValueError, AttributeError):
        return False

def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text with improved regex."""
    if not text:
        return []
        
    # More comprehensive email regex
    email_regex = r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'
    emails = re.findall(email_regex, text)
    
    # Additional validation
    valid_emails = []
    for email in emails:
        try:
            # Simple validation
            if '@' in email and '.' in email.split('@')[-1]:
                valid_emails.append(email.lower())  # Normalize to lowercase
        except (IndexError, AttributeError):
            continue
    
    return list(set(valid_emails))  # Remove duplicates

def extract_phone_numbers(text: str, country_code: str = None) -> List[str]:
    """Extract phone numbers with support for international formats."""
    if not text:
        return []
    
    # International phone number regex (simplified)
    phone_patterns = [
        r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',  # International
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US/Canada
    ]
    
    phones = []
    for pattern in phone_patterns:
        phones.extend(re.findall(pattern, text))
    
    # Clean and validate numbers
    cleaned = []
    for phone in phones:
        # Remove non-digit characters except leading +
        clean_phone = re.sub(r'(?<!^)[^\d+]', '', phone)
        
        # Add country code if missing and provided
        if country_code and not clean_phone.startswith('+'):
            clean_phone = f"+{country_code}{clean_phone}"
        
        # Basic validation
        if len(clean_phone) >= 8:  # Minimum length for a valid phone number
            cleaned.append(clean_phone)
    
    return list(set(cleaned))  # Remove duplicates

def json_serialize(obj: Any, **kwargs) -> str:
    """
    Enhanced JSON serialization with support for:
    - Dataclasses
    - Datetime objects
    - Sets
    - Custom serializers via kwargs
    """
    def default_serializer(o):
        if is_dataclass(o):
            return {k: v for k, v in asdict(o).items() 
                   if not k.startswith('_') and v is not None}
        elif hasattr(o, 'isoformat'):
            return o.isoformat()
        elif isinstance(o, set):
            return list(o)
        elif hasattr(o, '__dict__'):
            return {k: v for k, v in o.__dict__.items() 
                   if not k.startswith('_') and v is not None}
        elif callable(getattr(o, 'to_dict', None)):
            return o.to_dict()
        elif hasattr(o, 'tobytes'):  # Handle numpy arrays
            return o.tolist()
        return str(o)
    
    # Allow custom serializers via kwargs
    if 'default' not in kwargs:
        kwargs['default'] = default_serializer
    
    return json.dumps(obj, **kwargs)

@asynccontextmanager
async def get_session() -> aiohttp.ClientSession:
    """Get or create a shared aiohttp session with connection pooling."""
    global _SESSION
    
    if _SESSION is None or _SESSION.closed:
        # Configure timeout and connection limits
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(
            limit=100,  # Max simultaneous connections
            limit_per_host=10,  # Max connections per host
            enable_cleanup_closed=True,
            ssl=ssl.create_default_context()
        )
        
        # Set default headers
        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1',
        }
        
        _SESSION = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=connector,
            trust_env=True  # For proxy support
        )
    
    try:
        yield _SESSION
    except Exception as e:
        logger.error(f"Session error: {e}")
        if _SESSION and not _SESSION.closed:
            await _SESSION.close()
        _SESSION = None
        raise

async def fetch_with_retry(
    url: str,
    method: str = 'GET',
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (500, 502, 503, 504, 408, 429),
    **kwargs
) -> Optional[aiohttp.ClientResponse]:
    """
    Fetch a URL with retry logic using exponential backoff.
    
    Args:
        url: URL to fetch
        method: HTTP method (GET, POST, etc.)
        max_retries: Maximum number of retry attempts
        backoff_factor: Factor for exponential backoff
        status_forcelist: HTTP status codes that should trigger a retry
        **kwargs: Additional arguments to pass to aiohttp.ClientSession.request()
        
    Returns:
        aiohttp.ClientResponse if successful, None otherwise
    """
    if not url:
        raise ValueError("URL cannot be empty")
    
    # Set default timeout if not provided
    if 'timeout' not in kwargs:
        kwargs['timeout'] = aiohttp.ClientTimeout(total=30)
    
    # Set default headers if not provided
    headers = kwargs.pop('headers', {})
    if 'User-Agent' not in headers:
        headers['User-Agent'] = get_random_user_agent()
    
    # Configure retry strategy
    @backoff.on_exception(
        backoff.expo,
        (
            aiohttp.ClientError,
            aiohttp.ClientOSError,
            aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError,
            socket.gaierror,
            ssl.SSLError,
            ConnectionResetError,
        ),
        max_tries=max_retries + 1,  # +1 for the initial attempt
        factor=backoff_factor,
        jitter=backoff.full_jitter,
        max_time=300,  # Max total time in seconds
        on_backoff=lambda details: logger.warning(
            f"Retry {details['tries']}/{max_retries} for {url} "
            f"after {details['wait']:.2f}s: {details.get('exception', '')}"
        )
    )
    @backoff.on_predicate(
        backoff.expo,
        max_tries=max_retries + 1,
        factor=backoff_factor,
        jitter=backoff.full_jitter,
        max_time=300,
        on_backoff=lambda details: logger.warning(
            f"Retry {details['tries']}/{max_retries} for {url} "
            f"after {details['wait']:.2f}s: Status {details.get('value', {}).status}"
        )
    )
    async def _fetch() -> Optional[aiohttp.ClientResponse]:
        async with get_session() as session:
            try:
                response = await session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    **kwargs
                )
                
                # Check status code
                if response.status in status_forcelist:
                    response_text = await response.text(errors='ignore')
                    logger.warning(
                        f"HTTP {response.status} for {url}: {response_text[:200]}"
                    )
                    # Raise to trigger retry
                    response.raise_for_status()
                
                return response
                
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    logger.warning(f"Resource not found: {url}")
                    return None
                raise
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                raise
    
    try:
        return await _fetch()
    except Exception as e:
        logger.error(f"Failed to fetch {url} after {max_retries} retries: {e}")
        return None

async def fetch_robots_txt(base_url: str) -> dict:
    """
    Fetch and parse robots.txt for a given domain.
    
    Returns:
        dict: {
            'sitemaps': list of sitemap URLs,
            'rules': list of (user_agent, [disallowed_paths])
        }
    """
    if not base_url:
        return {'sitemaps': [], 'rules': []}
    
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    
    response = await fetch_with_retry(robots_url, max_retries=2)
    if not response:
        return {'sitemaps': [], 'rules': []}
    
    content = await response.text()
    
    # Parse robots.txt
    sitemaps = []
    rules = []
    current_user_agents = ['*']  # Default to all user agents
    
    for line in content.splitlines():
        line = line.split('#', 1)[0].strip()  # Remove comments
        if not line:
            continue
            
        try:
            directive, _, value = [x.strip() for x in line.partition(':')]
            directive = directive.lower()
            
            if directive == 'user-agent':
                current_user_agents = [ua.lower() for ua in value.split()]
            elif directive == 'disallow':
                for ua in current_user_agents:
                    rules.append((ua, value))
            elif directive == 'sitemap':
                sitemaps.append(value.strip())
        except (ValueError, IndexError):
            continue
    
    return {
        'sitemaps': sitemaps,
        'rules': rules
    }

async def parse_sitemap(sitemap_url: str) -> List[dict]:
    """Parse sitemap.xml and return list of URLs with metadata."""
    response = await fetch_with_retry(sitemap_url)
    if not response:
        return []
    
    content = await response.text()
    
    try:
        from xml.etree import ElementTree as ET
        
        # Handle sitemap index
        if 'sitemapindex' in content.lower():
            root = ET.fromstring(content)
            sitemaps = []
            
            for sitemap in root.findall('{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc is not None and loc.text:
                    sitemaps.append(loc.text)
            
            # Process all sitemaps in parallel
            tasks = [parse_sitemap(url) for url in sitemaps]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results
            return [item for sublist in results if isinstance(sublist, list) for item in sublist]
        
        # Handle regular sitemap
        elif 'urlset' in content.lower():
            root = ET.fromstring(content)
            urls = []
            
            for url in root.findall('{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                url_data = {}
                for child in url:
                    tag = child.tag.split('}')[-1]  # Remove namespace
                    url_data[tag] = child.text
                
                if 'loc' in url_data:
                    urls.append(url_data)
            
            return urls
        
        return []
    except Exception as e:
        logger.error(f"Error parsing sitemap {sitemap_url}: {e}")
        return []

async def get_site_urls(
    base_url: str, 
    max_urls: int = 1000,
    respect_robots: bool = True,
    include_sitemap: bool = True
) -> List[str]:
    """
    Get all URLs from a website using sitemap and crawling.
    
    Args:
        base_url: Base URL of the website
        max_urls: Maximum number of URLs to return
        respect_robots: Whether to respect robots.txt rules
        include_sitemap: Whether to include URLs from sitemap
        
    Returns:
        List of unique URLs
    """
    if not is_valid_url(base_url):
        raise ValueError(f"Invalid URL: {base_url}")
    
    urls = set()
    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
    
    # Get sitemap URLs
    sitemap_urls = []
    if include_sitemap:
        robots = await fetch_robots_txt(base_url)
        sitemap_urls.extend(robots['sitemaps'])
        
        # Try common sitemap locations if none found
        if not sitemap_urls:
            common_locations = [
                f"{base_domain}/sitemap.xml",
                f"{base_domain}/sitemap_index.xml",
                f"{base_domain}/sitemap/sitemap.xml",
                f"{base_domain}/sitemap-index.xml",
            ]
            
            # Check which sitemaps exist
            tasks = [
                fetch_with_retry(url, method='HEAD') 
                for url in common_locations
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for url, resp in zip(common_locations, responses):
                if isinstance(resp, aiohttp.ClientResponse) and resp.status == 200:
                    sitemap_urls.append(url)
    
    # Parse all sitemaps
    tasks = [parse_sitemap(url) for url in sitemap_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Add URLs from sitemaps
    for result in results:
        if isinstance(result, list):
            for item in result:
                if 'loc' in item and is_valid_url(item['loc']):
                    urls.add(normalize_url(item['loc']))
                    if len(urls) >= max_urls:
                        return list(urls)[:max_urls]
    
    # If we still don't have enough URLs, crawl the homepage for links
    if len(urls) < max_urls // 2:  # Only crawl if we have few URLs
        response = await fetch_with_retry(base_url)
        if response:
            content = await response.text()
            soup = BeautifulSoup(content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                url = normalize_url(link['href'], base_url)
                if is_valid_url(url) and urlparse(url).netloc == parsed_base.netloc:
                    urls.add(url)
                    if len(urls) >= max_urls:
                        break
    
    return list(urls)[:max_urls]
