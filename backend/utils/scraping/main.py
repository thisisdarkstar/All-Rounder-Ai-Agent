"""
Web Scraping Examples

This file demonstrates how to use the scraping utilities with practical examples.
Run this file directly to see the examples in action.
"""

import asyncio
import json
from pathlib import Path
from typing import List, Optional

# Import from the package
from backend.utils.scraping.search import search, search_images, SearchManager, SearchOptions
from backend.utils.scraping.fetcher import fetch_webpage, download_file, download_image
from backend.utils.scraping.models import SearchResult

async def example_web_search():
    """Example of performing a web search."""
    print("\n=== Example 1: Basic Web Search ===")
    query = "latest Python 3.12 features"
    print(f"Searching for: {query}")
    
    results = await search(query, max_results=3)
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   Source: {result.source}")
        if result.description:
            print(f"   {result.description[:150]}...")

async def example_image_search():
    """Example of searching for images."""
    print("\n=== Example 2: Image Search ===")
    query = "beautiful landscape"
    print(f"Searching for images of: {query}")
    
    images = await search_images(query, max_results=2)
    
    print(f"\nFound {len(images)} images:")
    for i, img in enumerate(images, 1):
        print(f"\n{i}. {img.title or 'Untitled Image'}")
        print(f"   Image URL: {img.url}")
        if img.thumbnail:
            print(f"   Thumbnail: {img.thumbnail}")

async def example_fetch_webpage():
    """Example of fetching and parsing a web page."""
    print("\n=== Example 3: Fetch Web Page Content ===")
    url = "https://httpbin.org/html"  # Using a test URL
    print(f"Fetching content from: {url}")
    
    page = await fetch_webpage(url)
    
    if page:
        print("\nPage Information:")
        print(f"Title: {page.title}")
        print(f"Status Code: {page.status_code}")
        print(f"Content Type: {page.content_type}")
        print(f"Content Preview: {page.content[:200]}...")
        print(f"Found {len(page.links)} links on the page")
    else:
        print("Failed to fetch the web page")

async def example_download_file():
    """Example of downloading a file."""
    print("\n=== Example 4: File Download ===")
    
    # Create a downloads directory if it doesn't exist
    download_dir = Path("downloads")
    download_dir.mkdir(exist_ok=True)
    
    # Example PDF URL (a sample PDF from the internet)
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    print(f"Downloading file from: {pdf_url}")
    
    file_path = await download_file(
        pdf_url,
        save_dir=download_dir,
        filename="sample_document.pdf"
    )
    
    if file_path:
        print(f"File downloaded successfully to: {file_path}")
        print(f"File size: {file_path.stat().st_size / 1024:.2f} KB")
    else:
        print("Failed to download the file")

async def example_advanced_search():
    """Example of advanced search with custom options."""
    print("\n=== Example 5: Advanced Search with Custom Options ===")
    
    # Create custom search options
    options = SearchOptions(
        max_retries=3,
        timeout=30,
        rate_limit_delay=2.0,  # 2 seconds between requests
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) MyCustomAgent/1.0"
    )
    
    # Initialize search manager with custom options
    async with SearchManager(engines=['duckduckgo', 'bing'], options=options) as manager:
        # Perform a search
        print("Searching for 'Python async programming'...")
        results = await manager.search("Python async programming", max_results=2)
        
        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Source: {result.source}")
            
            # Fetch the first search result's page content
            if i == 1:
                print("\nFetching page content for the first result...")
                page = await fetch_webpage(result.url)
                if page:
                    print(f"   Title: {page.title}")
                    print(f"   Description: {page.description[:150]}...")
                else:
                    print("   Failed to fetch page content")

async def main():
    """Run all examples."""
    try:
        await example_web_search()
        await example_image_search()
        await example_fetch_webpage()
        await example_download_file()
        await example_advanced_search()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Cleanup (close any open sessions, etc.)
        pass

if __name__ == "__main__":
    print("Starting web scraping examples...")
    asyncio.run(main())
    print("\nAll examples completed!")
