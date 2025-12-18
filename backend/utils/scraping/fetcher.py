"""
Web page and file fetching functionality.
"""
import os
import requests
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple
from urllib.parse import urlparse, urljoin
from datetime import datetime

from bs4 import BeautifulSoup
from tqdm import tqdm
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import WebPage, SearchResult
from .logger import get_logger

logger = get_logger(__name__)

# Default headers to mimic a browser
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'DNT': '1',
}

def create_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """
    Create a requests session with retry logic.
    
    Args:
        retries: Number of retries for failed requests
        backoff_factor: Backoff factor for retries
        
    Returns:
        Configured requests.Session instance
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def fetch_webpage(url: str, timeout: int = 30, headers: Optional[Dict[str, str]] = None) -> Optional[WebPage]:
    """
    Fetch a web page and extract relevant information with improved error handling.
    
    Args:
        url: URL of the web page to fetch
        timeout: Request timeout in seconds
        headers: Optional custom headers
        
    Returns:
        WebPage object with page content and metadata, or None if fetch failed
    """
    session = create_session()
    headers = headers or DEFAULT_HEADERS
    
    try:
        response = session.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else ''
        
        # Extract meta description
        description = ''
        meta_desc = soup.find('meta', attrs={'name': 'description'}) or \
                   soup.find('meta', attrs={'property': 'og:description'})
        if meta_desc and meta_desc.get('content'):
            description = meta_desc['content'].strip()
            
        # Extract main content (simplified - consider using libraries like trafilatura for better content extraction)
        main_content = ''
        main_tag = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_tag:
            main_content = main_tag.get_text(' ', strip=True)
        else:
            # Fallback to body if no main content found
            body = soup.find('body')
            if body:
                main_content = body.get_text(' ', strip=True)
        
        # Extract links
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(url, href)
            links.append(full_url)
        
        return WebPage(
            url=url,
            title=title,
            description=description,
            content=main_content,
            links=links,
            status_code=response.status_code,
            content_type=response.headers.get('content-type', '')
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {url}: {str(e)}")
        return None

def download_file(
    url: str, 
    save_dir: Union[str, Path] = None, 
    filename: str = None, 
    chunk_size: int = 8192,
    timeout: int = 60,
    headers: Optional[Dict[str, str]] = None
) -> Optional[Path]:
    """
    Download a file from a URL to the specified directory with improved error handling.
    
    Args:
        url: URL of the file to download
        save_dir: Directory to save the file (default: current directory)
        filename: Name to save the file as (default: extract from URL)
        chunk_size: Size of chunks to download at a time
        timeout: Request timeout in seconds
        headers: Optional custom headers
        
    Returns:
        Path to the downloaded file, or None if download failed
    """
    session = create_session()
    headers = headers or DEFAULT_HEADERS
    
    try:
        # Ensure save directory exists
        save_dir = Path(save_dir or '.').resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get filename from URL if not provided
        if not filename:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path) or 'downloaded_file'
        
        filepath = save_dir / filename
        
        # Stream the download to handle large files
        with session.get(url, stream=True, timeout=timeout, headers=headers) as response:
            response.raise_for_status()
            
            # Get total file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Initialize progress bar
            progress_bar = tqdm(
                total=total_size, 
                unit='iB', 
                unit_scale=True,
                desc=f"Downloading {filename}",
                leave=False
            )
            
            # Save file in chunks
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        size = f.write(chunk)
                        progress_bar.update(size)
            
            progress_bar.close()
            
            # Verify download if size is known
            if total_size != 0 and progress_bar.n != total_size:
                logger.error(f"Download incomplete: {progress_bar.n} bytes downloaded, expected {total_size}")
                filepath.unlink(missing_ok=True)
                return None
                
            logger.info(f"Successfully downloaded {filepath}")
            return filepath
            
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        # Clean up partially downloaded file if it exists
        if 'filepath' in locals() and filepath.exists():
            filepath.unlink(missing_ok=True)
        return None

def download_image(
    url: str, 
    save_dir: Union[str, Path] = None, 
    filename: str = None, 
    **kwargs
) -> Optional[Path]:
    """
    Download an image from a URL with proper image handling.
    
    Args:
        url: URL of the image to download
        save_dir: Directory to save the image
        filename: Optional filename (without extension)
        **kwargs: Additional arguments to pass to download_file()
        
    Returns:
        Path to the downloaded image, or None if download failed
    """
    # Extract file extension from URL if not provided in filename
    if not filename:
        parsed_url = urlparse(url)
        ext = os.path.splitext(parsed_url.path)[1].lower()
        if not ext or len(ext) > 5:  # If no extension or too long, use jpg as default
            ext = '.jpg'
        filename = f"image_{int(datetime.now().timestamp())}{ext}"
    
    # Ensure filename has an image extension
    if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']):
        filename += '.jpg'
    
    # Set default save directory if not provided
    if save_dir is None:
        save_dir = Path(__file__).parent / 'downloads' / 'images'
    
    return download_file(url, save_dir=save_dir, filename=filename, **kwargs)
