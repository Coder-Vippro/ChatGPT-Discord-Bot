import requests
import httpx
import json
import re
import time
import logging
import asyncio
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional, Tuple, Union
from src.config.config import GOOGLE_API_KEY, GOOGLE_CX
import tiktoken
from collections import namedtuple
from contextlib import contextmanager
from urllib.parse import urlparse
from playwright.async_api import async_playwright
import traceback
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Google Search Result structure
GoogleResult = namedtuple("GoogleResult", ["title", "link", "snippet", "date", "used_for_content"])

# Memory efficient cache decorator
@lru_cache(maxsize=32)
def get_encoding():
    """Cache the tokenizer to save memory"""
    return tiktoken.get_encoding("cl100k_base")

# Lightweight user-agent rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0'
]

def rotate_user_agent():
    """Rotate through user agents to avoid detection"""
    return USER_AGENTS[int(time.time()) % len(USER_AGENTS)]

def google_custom_search(query: str, num_results: int = 5, max_tokens: int = 4000) -> dict:
    """Performs Google Search and scrapes content from results."""
    try:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': query, 'num': min(num_results, 10)},
            timeout=10
        )
        response.raise_for_status()
        search_results = response.json()

        formatted_results = {'query': query, 'results': [], 'combined_content': ""}
        links = [item.get('link', '') for item in search_results.get('items', [])]

        # Use async scraper for better performance
        combined_content, used_links = asyncio.run(scrape_multiple_links_async(links, max_tokens))

        for item in search_results.get('items', []):
            formatted_results['results'].append(GoogleResult(
                title=item.get('title', ''),
                link=item.get('link', ''),
                snippet=item.get('snippet', ''),
                date=item.get('formattedDate', ''),
                used_for_content=item.get('link', '') in used_links
            )._asdict())

        formatted_results['combined_content'] = combined_content
        return formatted_results
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Google API request failed: {str(e)}")
        return {'error': 'Google search request failed'}

async def scrape_multiple_links_async(links: List[str], max_tokens: int) -> Tuple[str, set]:
    """Scrapes content from multiple web pages asynchronously while respecting the token limit."""
    combined_content = []
    used_links = set()
    token_count = 0
    enc = get_encoding()
    
    # Group links by domain to improve scraping success rate and reduce resource usage
    domain_links = {}
    for link in links:
        domain = urlparse(link).netloc
        if domain not in domain_links:
            domain_links[domain] = []
        domain_links[domain].append(link)
    
    # Process URLs - first try with httpx (lightweight)
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for domain, domain_links_list in domain_links.items():
            for link in domain_links_list:
                if token_count >= max_tokens:
                    break
                
                # Try with standard httpx request first (lightweight)
                try:
                    headers = {"User-Agent": rotate_user_agent()}
                    response = await client.get(link, headers=headers)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Extract visible text
                        for script in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                            script.decompose()
                            
                        text = extract_main_content(soup)
                        tokens = len(enc.encode(text))
                        
                        if token_count + tokens <= max_tokens:
                            combined_content.append(text)
                            used_links.add(link)
                            token_count += tokens
                            continue  # Success with httpx, move to next URL
                
                except Exception as e:
                    logger.info(f"Standard request for {link} failed, trying with Playwright: {str(e)}")
                
                # If httpx fails or content is suspect bot detection, try with Playwright
                if token_count >= max_tokens:
                    break
                    
                try:
                    text = await scrape_with_playwright(link)
                    if text:
                        tokens = len(enc.encode(text))
                        if token_count + tokens <= max_tokens:
                            combined_content.append(text)
                            used_links.add(link)
                            token_count += tokens
                except Exception as e:
                    logger.error(f"❌ Failed to scrape {link}: {str(e)}")
                    traceback.print_exc()

    return " ".join(combined_content), used_links

def extract_main_content(soup):
    """Extract main content from webpage using heuristics for better quality"""
    # Find content areas by common tags and attributes
    main_content = soup.find('main') or soup.find('article') or soup.find(attrs={'class': re.compile('content|article|post', re.I)})
    
    if main_content:
        # Extract text from main content
        return ' '.join(main_content.stripped_strings)
    
    # Fallback to body content if no main content found
    return ' '.join(soup.stripped_strings)

async def scrape_with_playwright(url):
    """Use Playwright with stealth mode for websites that block bots"""
    try:
        async with async_playwright() as p:
            # Launch browser with minimal resources
            browser = await p.firefox.launch(
                headless=True,
                firefox_user_prefs={
                    "media.autoplay.default": 5,
                    "media.autoplay.blocking_policy": 1,
                    "media.navigator.mediacapabilities.enabled": False,
                    "permissions.default.image": 2,
                },
            )
            
            # Use Firefox context with stealth settings
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=rotate_user_agent(),
                bypass_csp=True,
                ignore_https_errors=True
            )
            
            # Add stealth mode
            await context.add_init_script("""
            // Stealth mode
            Object.defineProperty(navigator, 'webdriver', { get: () => false });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            """)
            
            page = await context.new_page()
            
            # Set memory-friendly timeout
            await page.goto(url, wait_until='domcontentloaded', timeout=15000)
            
            # Let page render a bit
            await asyncio.sleep(1)
            
            # Get page content
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract visible text
            for script in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                script.decompose()
                
            text = extract_main_content(soup)
            
            await browser.close()
            return text
    except Exception as e:
        logger.error(f"Playwright error on {url}: {str(e)}")
        return None
